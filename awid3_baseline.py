#!/usr/bin/env python3
"""
AWID3 baseline: Random Forest + small neural network (PyTorch) for benign-vs-attack detection.
Tested on real AWID3-CSV pieces (Deauth_22.csv, Evil_Twin_40.csv): all pieces share the same
254-column schema and a 'Label' column (Normal / Deauth / Evil_Twin).

HOW TO USE
  1. Put the downloaded 50,000-row CSV pieces in one folder, e.g. data/  (Deauth_*.csv, Evil_Twin_*.csv)
  2. python awid3_baseline.py --data data/ --out artifacts/
     Examples:
        python awid3_baseline.py --data data/ --out artifacts/                  # everything in data/
        python awid3_baseline.py --data data/ --split random                    # optimistic split
        python awid3_baseline.py --data data/ --per-group 12 --min-attack-rows 500   # 12 pieces per attack, only ones with attacks
        python awid3_baseline.py --data "data/Deauth_*.csv" "data/Evil_Twin_*.csv" --max-files 20

WHAT IT DOES
  load pieces -> binary label (Normal=0, any attack=1) -> drop leaky/identifier columns ->
  keep numeric columns -> block-wise train/val/test split -> scale -> Random Forest + MLP ->
  accuracy/precision/recall/F1/confusion matrix, plus recall for each attack type.

OUTPUT (in --out): metrics.json, feature_importance_rf.csv, features.json, scaler.joblib,
                   rf.joblib, mlp.pt, test_set.npz (scaled X_test, y_test, attack_type),
                   sentinel.json (fill values), run_config.json
"""
import argparse
import json
import os
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from awid3_common import (binary_target, drop_bad_tls_rows, expand_paths,
                          find_label_col, piece_label_counts, read_awid3, to_numeric_series)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ----------------------------- SETTINGS (edit me) -----------------------------
DATA_PATHS = ["data/"]            # files, folders, or glob patterns
LABEL_COL = None                  # None = auto-detect ('Label' in the real files)
MAX_ROWS_PER_FILE = 50000         # rows read from each CSV (a piece has 50,000)
MAX_FILES = None                  # e.g. 10 to cap how many files are loaded
BLOCK_SIZE = 2500                 # rows per block for the block-wise split
# Columns dropped before training (checked against the real Deauth and Evil_Twin pieces):
#  * absolute time / counters (frame.number, frame.time*, mactime, tsf, timestamps): attack frames
#    sit in one contiguous stretch of each capture (e.g. frames 2,000,002-2,049,796 in Evil_Twin_40),
#    so these would leak the label. frame.time_delta (gap between frames) is kept: it is behavior
#    and attacker-controllable.
#  * addresses, SSIDs, keys, payloads, ids, TCP/UDP ports and sequence numbers: identify devices or
#    sessions rather than behavior. (SSID/BSSID removal means the model cannot just spot "same SSID,
#    different BSSID"; it has to learn behavior such as signal, rate, length and timing.)
DROP_REGEX = (r"(^frame\.number$|^frame\.time$|^frame\.time_epoch$|^frame\.time_relative$|"
              r"^frame\.time_delta_displayed$|^frame\.encap_type$|mactime|timestamp|_tsf$|"
              r"\.time_relative$|\.sa$|\.da$|\.ta$|\.ra$|\.src$|\.dst$|bssid|ssid|\.addr|"
              r"mac_addr|hw_mac|payload|\.data$|data_data|uuid|cookie|\.seq$|\.seq_raw$|\.ack_raw$|"
              r"\.ack$|fcs|\.id$|\.nonce$|\.key$|pmkid|\.kck$|\.kek$|key_mic|replay_counter|\.sesid$|"
              r"\.tid$|\.pid|\.mid$|msg_id|\.fid$|connection_id|port$|\.checksum$)")
MAX_MISSING = 0.999               # most AWID3 columns are empty for most frames; keep sparse ones
MIN_NONNULL = 50                  # ...but a column needs at least this many real values
MIN_NUMERIC_SHARE = 0.9           # a column must be >=90% numeric after conversion
SEED = 42
# ------------------------------------------------------------------------------


def reduce_frame(df, label_col):
    """Shrink one 254-column piece to its numeric, non-identifier columns right after reading it.
    Without this, loading hundreds of pieces would keep ~250 text columns each in memory."""
    drop_re = re.compile(DROP_REGEX, re.I)
    keep = {}
    for c in df.columns:
        if c == label_col or drop_re.search(c):
            continue
        s = df[c]
        if s.notna().sum() == 0:
            continue                      # empty in this piece; absent columns become NaN on concat
        t = to_numeric_series(s)
        if t.notna().sum() / s.notna().sum() < MIN_NUMERIC_SHARE:
            continue
        keep[c] = t.astype("float32")
    out = pd.DataFrame(keep, index=df.index)
    out["_label"] = df[label_col].astype(str).str.strip().values
    return out


def has_attack_rows(f, min_rows):
    counts = piece_label_counts(f)
    attack = sum(v for k, v in counts.items() if binary_target(pd.Series([k])).iloc[0] == 1)
    return attack >= min_rows


def pick_files(files, max_files, per_group, min_attack_rows=0):
    """per_group = K: take K evenly spaced pieces from each attack group (Deauth_*, Evil_Twin_*, ...),
    so a small sample still covers the whole capture and every attack type."""
    if min_attack_rows:
        n0 = len(files)
        files = [f for f in files if has_attack_rows(f, min_attack_rows)]
        print(f"  pieces with >= {min_attack_rows} attack rows: {len(files)} of {n0}")
    if per_group:
        groups = {}
        for f in files:
            key = re.sub(r"[_\-\s]*\d+$", "", os.path.splitext(os.path.basename(f))[0])
            groups.setdefault(key, []).append(f)
        files = []
        for key, fs in sorted(groups.items()):
            fs = sorted(fs, key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)])
            idx = np.unique(np.linspace(0, len(fs) - 1, min(per_group, len(fs))).round().astype(int))
            files += [fs[i] for i in idx]
            print(f"  group {key!r}: {len(fs)} pieces, using {len(idx)}")
    if max_files:
        files = files[:max_files]
    return files


def load_data(paths, max_rows, max_files, label_col, per_group=None, min_attack_rows=0):
    files = pick_files(expand_paths(paths), max_files, per_group, min_attack_rows)
    frames = []
    for f in files:
        raw = read_awid3(f, nrows=max_rows)
        lc = label_col or find_label_col(raw)
        raw = drop_bad_tls_rows(raw)
        df = reduce_frame(raw, lc)
        df["_source_file"] = os.path.basename(f)
        df["_row"] = np.arange(len(df), dtype="int32")
        frames.append(df)
        n_att = int((binary_target(df["_label"]) == 1).sum())
        print(f"  loaded {os.path.basename(f)}: {len(df):,} rows, {n_att:,} attack")
        del raw
    df = pd.concat(frames, ignore_index=True, sort=False)
    print("Labels:\n" + df["_label"].value_counts(dropna=False).head(20).to_string())
    return df, "_label"


def build_features(df, label_col):
    df = df.reset_index(drop=True)
    y = binary_target(df[label_col]).to_numpy()
    attack_type = np.where(y == 1, df[label_col].astype(str).str.strip(), "Normal")
    groups = (df["_source_file"].astype(str) + "#" + (df["_row"] // BLOCK_SIZE).astype(str)).to_numpy()
    X = df.drop(columns=[label_col, "_source_file", "_row"], errors="ignore")
    keep = [c for c in X.columns
            if X[c].isna().mean() <= MAX_MISSING and X[c].notna().sum() >= MIN_NONNULL
            and X[c].nunique(dropna=True) > 1]
    Xn = X[keep].replace([np.inf, -np.inf], np.nan)
    print(f"Features kept: {len(keep)} of {X.shape[1]}  |  rows: {len(Xn):,}  "
          f"|  attack share: {y.mean():.3f}")
    return Xn, y, attack_type, groups


def split_data(X, y, groups, mode):
    """random = packet-level (optimistic: neighbouring packets can land on both sides)
       blocks = whole blocks of consecutive packets go to one side only (more honest)."""
    idx = np.arange(len(y))
    if mode == "random":
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
        return tr, va, te
    for attempt in range(50):  # re-draw until every split has both classes
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED + attempt)
        tr, tmp = next(gss.split(idx, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + attempt)
        va_i, te_i = next(gss2.split(tmp, y[tmp], groups[tmp]))
        va, te = tmp[va_i], tmp[te_i]
        if all(len(np.unique(y[s])) == 2 for s in (tr, va, te)):
            return tr, va, te
    raise SystemExit("Could not build block splits that contain both classes in every split. "
                     "Load more files, or use --split random.")


def metrics(y_true, y_pred):
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist())


def per_attack_recall(attack_type, y_pred):
    out = {}
    for t in sorted(set(attack_type) - {"Normal"}):
        m = attack_type == t
        out[t] = dict(n=int(m.sum()), detected=int(y_pred[m].sum()),
                      recall=float(y_pred[m].mean()))
    m = attack_type == "Normal"
    out["Normal"] = dict(n=int(m.sum()), false_alarms=int(y_pred[m].sum()),
                         false_alarm_rate=float(y_pred[m].mean()))
    return out


class MLP(nn.Module):
    def __init__(self, n_in, h1=128, h2=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xva, yva, epochs, batch, lr, patience, device):
    torch.manual_seed(SEED)
    model = MLP(Xtr.shape[1]).to(device)
    pos = max(ytr.sum(), 1)
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), yva_t).item()
        print(f"  epoch {ep:3d}  train loss {tot / len(perm):.4f}  val loss {vloss:.4f}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("  early stopping")
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict_mlp(model, X, device):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
    return (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DATA_PATHS)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--label-col", default=LABEL_COL)
    ap.add_argument("--max-rows-per-file", type=int, default=MAX_ROWS_PER_FILE)
    ap.add_argument("--max-files", type=int, default=MAX_FILES)
    ap.add_argument("--min-attack-rows", type=int, default=0,
                    help="only use pieces with at least this many attack rows (e.g. 500)")
    ap.add_argument("--per-group", type=int, default=None,
                    help="use K evenly spaced pieces from each attack group (e.g. 12)")
    ap.add_argument("--split", choices=["blocks", "random"], default="blocks")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--rf-trees", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df, label_col = load_data(a.data, a.max_rows_per_file, a.max_files, a.label_col, a.per_group, a.min_attack_rows)
    X, y, attack_type, groups = build_features(df, label_col)
    if len(np.unique(y)) < 2:
        raise SystemExit("Only one class present. Load both normal and attack files.")

    tr, va, te = split_data(X, y, groups, a.split)
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte, att_te = y[tr], y[va], y[te], attack_type[te]
    # A missing value means "this frame type has no such field" (e.g. no reason code on a beacon),
    # which is informative, so fill with a value just below the column minimum instead of the median.
    sentinel = (Xtr.min() - 1.0).fillna(-1.0)
    Xtr, Xva, Xte = (d.fillna(sentinel) for d in (Xtr, Xva, Xte))
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = (scaler.transform(d) for d in (Xtr, Xva, Xte))
    print(f"Split ({a.split}): train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}  "
          f"| attack share train {ytr.mean():.3f} / test {yte.mean():.3f}")

    results = {"split": a.split, "n_features": X.shape[1],
               "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte))}
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=a.rf_trees, n_jobs=-1, random_state=SEED,
                                class_weight="balanced_subsample").fit(Xtr_s, ytr)
    p = rf.predict(Xte_s)
    results["random_forest"] = metrics(yte, p)
    results["random_forest"]["per_attack"] = per_attack_recall(att_te, p)
    print(f"Random Forest done in {time.time() - t0:.1f}s: "
          f"{ {k: round(v, 4) for k, v in results['random_forest'].items() if isinstance(v, float)} }")
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(a.out, "feature_importance_rf.csv"), header=["importance"])
    print("Top RF features:\n" + imp.head(10).round(4).to_string())
    if imp.iloc[0] > 0.5:
        print(f"WARNING: '{imp.index[0]}' carries {imp.iloc[0]:.0%} of the importance. "
              "Check it is real behavior and not a capture artifact or label leak.")

    t0 = time.time()
    mlp = train_mlp(Xtr_s, ytr, Xva_s, yva, a.epochs, a.batch, a.lr, a.patience, device)
    p = predict_mlp(mlp, Xte_s, device)
    results["mlp"] = metrics(yte, p)
    results["mlp"]["per_attack"] = per_attack_recall(att_te, p)
    print(f"MLP done in {time.time() - t0:.1f}s")

    joblib.dump(rf, os.path.join(a.out, "rf.joblib"))
    joblib.dump(scaler, os.path.join(a.out, "scaler.joblib"))
    torch.save(mlp.state_dict(), os.path.join(a.out, "mlp.pt"))
    json.dump(list(X.columns), open(os.path.join(a.out, "features.json"), "w"), indent=1)
    json.dump({k: float(v) for k, v in sentinel.items()}, open(os.path.join(a.out, "sentinel.json"), "w"))
    json.dump(vars(a), open(os.path.join(a.out, "run_config.json"), "w"), indent=1)
    json.dump(results, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "test_set.npz"), X_test=Xte_s, y_test=yte,
                        attack_type=att_te)
    print(f"\nSaved results to {a.out}/\n")
    print(f"{'model':14s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
    for k in ("random_forest", "mlp"):
        v = results[k]
        print(f"{k:14s} {v['accuracy']:7.4f} {v['precision']:7.4f} {v['recall']:7.4f} {v['f1']:7.4f}")
        for t, d in v["per_attack"].items():
            if t == "Normal":
                print(f"    {t:10s} false-alarm rate {d['false_alarm_rate']:.4f} (n={d['n']:,})")
            else:
                print(f"    {t:10s} recall {d['recall']:.4f} (n={d['n']:,})")


if __name__ == "__main__":
    main()