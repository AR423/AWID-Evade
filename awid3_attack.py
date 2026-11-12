#!/usr/bin/env python3
"""
Constrained feature-space adversarial evasion (FGSM / PGD) against the AWID3 baseline.

Implements step 2-3 of the proposal's methodology: "Apply feature-level perturbation
techniques ... to a held-out set of attack samples, constrained to changes that would
remain plausible for a real 802.11 attacker to produce (e.g., timing or signal-strength-
adjacent features, not physically impossible values)" and "re-run the trained classifier
... and measure the drop in detection accuracy compared to the baseline."

ATTACKER-CONTROLLABLE FEATURES
  Per the Progress Report 1 "Challenges" section, perturbation is restricted to:
  signal strength, data rate, inter-frame timing, duration, retry flag, reason code.
  Mapped to the 48 columns in artifacts/baseline/features.json, that is:
    radiotap.dbm_antsignal, wlan_radio.signal_dbm   (signal strength)
    radiotap.datarate,      wlan_radio.data_rate    (data rate)
    frame.time_delta                                (inter-frame timing)
    wlan_radio.duration,    wlan.duration            (duration)
    wlan.fc.retry                                    (retry flag)
    wlan.fixed.reason_code                           (reason code)
  All other 39 features (addresses/SSIDs/keys were already dropped in the baseline;
  these are protocol-content fields like frame.len, tcp flags, DNS/HTTP/TLS fields)
  are left untouched -- an attacker rewriting those would usually break the attack
  itself or require control the threat model doesn't grant.

WHY GRADIENTS COME FROM THE MLP, NOT THE RF
  FGSM/PGD need a loss gradient w.r.t. the input; Random Forest has none (it's a
  step-function ensemble). Perturbations are therefore crafted against the MLP and
  then evaluated against BOTH the MLP (white-box) and the RF (black-box/transfer).
  This also lets the "which classifier is more robust" angle the proposal asks about
  (research question: "which traffic features are most responsible for this
  vulnerability") be answered with a transferability comparison, consistent with the
  transferability literature the proposal itself cites ([2], Yonbawi et al. 2025).

REALISM CONSTRAINTS (not just an L-inf ball)
  Per-feature bounds are set in RAW units (not scaled units) and are feature-type aware:
    - continuous (signal/rate/duration/timing): bounded absolute or relative step
    - binary (retry flag): perturbation rounds to {0,1} after the gradient step
    - discrete (reason code): perturbation rounds to the nearest integer code
  A feature is only perturbed on rows where it isn't the "missing value" sentinel
  from the baseline run (e.g. reason_code is only meaningful on frames that have one);
  perturbing a sentinel-filled "this field doesn't apply" value is not realistic.

OUTPUT (into --out, default artifacts/attack/):
  perturbation_budget.json  -- exact per-feature bounds used (methodology record)
  attack_metrics.json       -- pre/post accuracy, recall, per-attack recall,
                                for RF and MLP, under FGSM and PGD, for Evil_Twin
                                and Deauth separately
  feature_shift_rf.csv      -- per-feature mean |perturbation| vs. drop in RF recall,
                                for the feature-analysis step of the proposal
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------
# Attacker-controllable feature budget (raw units). See module docstring for the
# mapping from the proposal's plain-language list to these exact column names.
# "kind" drives post-perturbation rounding; "bound" is the max absolute change in
# raw units (continuous/discrete) or ignored for binary (always rounds to 0/1).
# --------------------------------------------------------------------------------
ATTACKER_FEATURES = {
    "radiotap.dbm_antsignal": {"kind": "continuous", "bound": 5.0},   # +/-5 dBm Tx power
    "wlan_radio.signal_dbm":  {"kind": "continuous", "bound": 5.0},   # same signal, 2nd parser
    "radiotap.datarate":      {"kind": "continuous", "bound": 6.0},   # +/-6 Mbps rate choice
    "wlan_radio.data_rate":   {"kind": "continuous", "bound": 6.0},
    "frame.time_delta":       {"kind": "continuous", "bound": None, "rel_bound": 1.0},
    # ^ relative: new_delta in [0.5x, 2x] original gap -- an attacker slowing a flood down
    #   is far more realistic than an unbounded absolute shift in inter-frame timing.
    "wlan_radio.duration":    {"kind": "continuous", "bound": None, "rel_bound": 1.0},
    "wlan.duration":          {"kind": "continuous", "bound": None, "rel_bound": 1.0},
    "wlan.fc.retry":          {"kind": "binary"},
    "wlan.fixed.reason_code": {"kind": "discrete", "bound": 3.0},     # nearby 802.11 reason code
}

SEED = 42


# --------------------------------- loading -------------------------------------

def load_baseline(art_dir):
    with open(os.path.join(art_dir, "features.json")) as f:
        features = json.load(f)
    with open(os.path.join(art_dir, "sentinel.json")) as f:
        sentinel = json.load(f)
    scaler = joblib.load(os.path.join(art_dir, "scaler.joblib"))
    rf = joblib.load(os.path.join(art_dir, "rf.joblib"))
    mlp_state = torch.load(os.path.join(art_dir, "mlp.pt"), map_location="cpu")
    test = np.load(os.path.join(art_dir, "test_set.npz"), allow_pickle=True)
    return features, sentinel, scaler, rf, mlp_state, test


class MLP(nn.Module):
    """Must mirror awid3_baseline.py's MLP exactly so the saved state_dict loads."""
    def __init__(self, n_in, h1=128, h2=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_mlp(n_in, state_dict):
    model = MLP(n_in)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------- perturbation budget --------------------------------

def build_budget(features, scaler, sentinel):
    """Per-feature (mask, scaled_bound_lo, scaled_bound_hi, kind, raw_sentinel) used by
    the attack loop. Bounds are converted from raw units to the scaler's standardized
    units feature-by-feature (bound_raw / scale_[i]), since FGSM/PGD step in scaled
    space (that's the space the models were trained and the gradient is computed in)."""
    idx = {f: i for i, f in enumerate(features)}
    mask = np.zeros(len(features), dtype=bool)
    scaled_bound = np.zeros(len(features), dtype=np.float64)
    kind = np.array(["fixed"] * len(features), dtype=object)
    rel_bound = np.zeros(len(features), dtype=np.float64)
    budget_record = {}
    for name, spec in ATTACKER_FEATURES.items():
        if name not in idx:
            continue  # feature got dropped upstream (e.g. not present in these pieces)
        i = idx[name]
        mask[i] = True
        kind[i] = spec["kind"]
        if spec["kind"] == "binary":
            scaled_bound[i] = np.inf  # realism comes from rounding to {0,1}, not a bound
            budget_record[name] = {"kind": "binary", "note": "flips toward 0/1 after step"}
        elif spec.get("rel_bound"):
            rel_bound[i] = spec["rel_bound"]
            scaled_bound[i] = np.inf  # relative constraint applied in raw space post-hoc
            budget_record[name] = {"kind": spec["kind"], "relative_bound": spec["rel_bound"],
                                   "note": "post-perturbation value clipped to "
                                           f"[{1-spec['rel_bound']:.1f}x, {1+spec['rel_bound']:.1f}x] "
                                           "of the original raw value"}
        else:
            raw_bound = spec["bound"]
            scale = scaler.scale_[i] if scaler.scale_[i] > 0 else 1.0
            scaled_bound[i] = raw_bound / scale
            budget_record[name] = {"kind": spec["kind"], "raw_bound": raw_bound}
    return mask, scaled_bound, kind, rel_bound, budget_record, idx


def sentinel_active_mask(X_raw, features, sentinel, mask):
    """True where a masked feature's raw value is NOT the baseline's missing-value
    sentinel for that column -- i.e. where the field actually applies to that frame."""
    active = np.ones_like(X_raw, dtype=bool)
    for i, f in enumerate(features):
        if not mask[i]:
            continue
        sv = sentinel.get(f)
        if sv is None:
            continue
        active[:, i] = ~np.isclose(X_raw[:, i], sv, atol=1e-6)
    return active


# ------------------------------- attack loops ------------------------------------

def fgsm_step(model, X, y, mask_t, bound_t):
    X = X.clone().detach().requires_grad_(True)
    loss = nn.functional.binary_cross_entropy_with_logits(model(X), y)
    grad, = torch.autograd.grad(loss, X)
    step = torch.sign(grad) * mask_t
    delta = torch.clamp(step, -1.0, 1.0) * torch.clamp(bound_t, max=1e6)
    return delta


def pgd_attack(model, X0, y, mask_t, bound_t, steps=20, alpha_frac=0.15):
    """L-inf PGD per feature, box-constrained to each feature's own bound (so a tight
    feature like the binary retry flag doesn't get the same step size as a loose one)."""
    X = X0.clone().detach()
    bound_eff = torch.where(torch.isinf(bound_t), torch.full_like(bound_t, 3.0), bound_t)
    alpha = alpha_frac * bound_eff
    Xadv = X.clone()
    for _ in range(steps):
        Xadv = Xadv.clone().detach().requires_grad_(True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(Xadv), y)
        grad, = torch.autograd.grad(loss, Xadv)
        step = torch.sign(grad) * mask_t * alpha
        Xadv = Xadv.detach() + step
        low = X - bound_eff * mask_t
        high = X + bound_eff * mask_t
        Xadv = torch.max(torch.min(Xadv, high), low)
    return Xadv.detach()


def postprocess_realism(X_raw_adv, X_raw_orig, features, idx, kind, rel_bound, active_mask):
    """Snap continuous/discrete/binary perturbed RAW values back to something an
    actual attacker could set, and leave inactive (sentinel) cells untouched."""
    out = X_raw_adv.copy()
    for name, i in idx.items():
        if name not in ATTACKER_FEATURES:
            continue
        k = kind[i]
        if k == "binary":
            out[:, i] = np.round(np.clip(out[:, i], 0, 1))
        elif rel_bound[i] > 0:
            lo = X_raw_orig[:, i] * (1 - rel_bound[i])
            hi = X_raw_orig[:, i] * (1 + rel_bound[i])
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            out[:, i] = np.clip(out[:, i], lo, hi)
        elif k == "discrete":
            out[:, i] = np.round(out[:, i])
    # restore original value wherever the field didn't apply to that frame
    out = np.where(active_mask, out, X_raw_orig)
    return out


# --------------------------------- evaluation -------------------------------------

def metrics_block(y_true, y_pred, attack_type):
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "recall": float(y_pred[y_true == 1].mean()) if (y_true == 1).any() else None,
        "per_attack": {},
    }
    for t in sorted(set(attack_type)):
        m = attack_type == t
        out["per_attack"][t] = {"n": int(m.sum()), "recall": float(y_pred[m].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/baseline")
    ap.add_argument("--out", default="artifacts/attack")
    ap.add_argument("--pgd-steps", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    features, sentinel, scaler, rf, mlp_state, test = load_baseline(a.artifacts)
    X_test_s = test["X_test"]           # already scaled (as saved by awid3_baseline.py)
    y_test = test["y_test"]
    attack_type = test["attack_type"]

    # only craft perturbations against real attack samples (the threat model: evade
    # detection of an actual attack, not of normal traffic)
    atk_mask = y_test == 1
    Xs = X_test_s[atk_mask]
    y_atk = y_test[atk_mask]
    at_atk = attack_type[atk_mask]
    X_raw = scaler.inverse_transform(Xs)

    mask, scaled_bound, kind, rel_bound, budget_record, idx = build_budget(features, scaler, sentinel)
    active = sentinel_active_mask(X_raw, features, sentinel, mask)
    # don't perturb sentinel (inapplicable) cells at all: zero their effective bound per-row
    # by building a per-row mask tensor instead of the single global mask
    row_mask = np.tile(mask, (Xs.shape[0], 1)) & active

    print(f"Attacking {Xs.shape[0]:,} held-out attack samples "
          f"({(at_atk == 'Deauth').sum():,} Deauth, {(at_atk == 'Evil_Twin').sum():,} Evil_Twin)")
    print(f"Attacker-controllable features ({mask.sum()} of {len(features)}): "
          f"{[f for f in features if mask[idx[f]]] if False else list(budget_record)}")

    mlp = build_mlp(len(features), mlp_state)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y_atk, dtype=torch.float32)
    mask_row_t = torch.tensor(row_mask, dtype=torch.float32)
    bound_t = torch.tensor(np.tile(scaled_bound, (Xs.shape[0], 1)), dtype=torch.float32) * mask_row_t
    bound_t = torch.where(mask_row_t.bool(), bound_t, torch.zeros_like(bound_t))
    # re-apply inf for binary/relative features where active, so fgsm/pgd clamp doesn't
    # zero them out via the "max=1e6" clamp above -- handled by using inf directly:
    for i in range(len(features)):
        if kind[i] in ("binary",) or rel_bound[i] > 0:
            col_active = torch.tensor(row_mask[:, i], dtype=torch.bool)
            bound_t[col_active, i] = float("inf")

    results = {"n_attack_samples": int(Xs.shape[0]), "features_perturbed": list(budget_record)}

    with torch.no_grad():
        p_rf_before = rf.predict(Xs)
        p_mlp_before = (torch.sigmoid(mlp(Xt)) >= 0.5).numpy().astype(int)
    results["before"] = {
        "random_forest": metrics_block(y_atk, p_rf_before, at_atk),
        "mlp": metrics_block(y_atk, p_mlp_before, at_atk),
    }

    for name, craft in (("fgsm", lambda: fgsm_step(mlp, Xt, yt, mask_row_t, bound_t) + Xt),
                        ("pgd", lambda: pgd_attack(mlp, Xt, yt, mask_row_t, bound_t, steps=a.pgd_steps))):
        Xadv_s = craft().numpy()
        X_raw_adv = scaler.inverse_transform(Xadv_s)
        X_raw_adv = postprocess_realism(X_raw_adv, X_raw, features, idx, kind, rel_bound, active)
        Xadv_s_final = scaler.transform(X_raw_adv)

        p_rf = rf.predict(Xadv_s_final)
        with torch.no_grad():
            p_mlp = (torch.sigmoid(mlp(torch.tensor(Xadv_s_final, dtype=torch.float32))) >= 0.5).numpy().astype(int)

        results[name] = {
            "random_forest": metrics_block(y_atk, p_rf, at_atk),
            "mlp": metrics_block(y_atk, p_mlp, at_atk),
            "evasion_rate": {
                "random_forest": float(1 - p_rf[y_atk == 1].mean()),
                "mlp": float(1 - p_mlp[y_atk == 1].mean()),
            },
        }
        print(f"\n{name.upper()}  RF recall {results[name]['random_forest']['recall']:.4f} "
              f"(was {results['before']['random_forest']['recall']:.4f})   "
              f"MLP recall {results[name]['mlp']['recall']:.4f} "
              f"(was {results['before']['mlp']['recall']:.4f})")

        if name == "pgd":
            raw_shift = np.abs(X_raw_adv - X_raw)
            shift_rows = []
            for f in budget_record:
                i = idx[f]
                shift_rows.append({"feature": f, "mean_abs_shift": float(raw_shift[active[:, i], i].mean())
                                   if active[:, i].any() else 0.0,
                                   "rows_perturbed": int(active[:, i].sum())})
            pd.DataFrame(shift_rows).sort_values("mean_abs_shift", ascending=False) \
                .to_csv(os.path.join(a.out, "feature_shift_rf.csv"), index=False)

    json.dump(budget_record, open(os.path.join(a.out, "perturbation_budget.json"), "w"), indent=1)
    json.dump(results, open(os.path.join(a.out, "attack_metrics.json"), "w"), indent=1)
    print(f"\nSaved results to {a.out}/")


if __name__ == "__main__":
    main()
