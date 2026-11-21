#!/usr/bin/env python3
"""
awid3_extract.py -- turn a pcap recaptured by awid3_inject.py into a feature
row with the exact same columns, in the exact same order, that the trained
AWID3 models expect (features.json), then score it with the saved scaler,
Random Forest, and MLP from a baseline run.

AWID3's CSV feature names ARE Wireshark/tshark field names (frame.len,
radiotap.dbm_antsignal, wlan.fc.retry, ...), so tshark can re-derive a matching
feature table directly from our own recaptured pcap with no custom parser.

Run this on the SAME machine/environment where awid3_baseline.py's output
lives (scaler.joblib, rf.joblib, mlp.pt, features.json, sentinel.json), with
tshark installed and this repo's awid3_common.py / awid3_baseline.py importable.

Usage (both clean and perturbed Deauth frames are ground-truth attacks, so
pass --label 1 for both -- the "recall" printed is how many of them the model
still catches; --label 0 would report a false-alarm rate instead):

    python3 awid3_extract.py --pcap clean.pcap     --label 1 --artifacts artifacts/run_all2
    python3 awid3_extract.py --pcap perturbed.pcap --label 1 --artifacts artifacts/run_all2
"""
import argparse
import json
import subprocess
import sys
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from awid3_common import to_numeric_series
from awid3_baseline import MLP


def tshark_extract(pcap, fields):
    args = ["tshark", "-r", pcap, "-T", "fields"]
    for f in fields:
        args += ["-e", f]
    args += ["-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return pd.read_csv(StringIO(out), dtype=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--label", type=int, choices=[0, 1], required=True)
    ap.add_argument("--artifacts", default="artifacts/run_all2",
                     help="folder holding features.json/sentinel.json/scaler.joblib/rf.joblib/mlp.pt")
    a = ap.parse_args()

    features = json.load(open(f"{a.artifacts}/features.json"))
    sentinel = json.load(open(f"{a.artifacts}/sentinel.json"))

    raw = tshark_extract(a.pcap, features)
    if "wlan.fc.subtype" in raw.columns:
        raw = raw[raw["wlan.fc.subtype"].astype(str).isin(["12", "0xc", "0x0c"])]
    if raw.empty:
        raise SystemExit("No Deauth frames found in the pcap -- check the capture/injection.")

    X = pd.DataFrame({c: (to_numeric_series(raw[c]) if c in raw.columns
                          else pd.Series(np.nan, index=raw.index))
                       for c in features})
    X = X.fillna(pd.Series(sentinel))

    scaler = joblib.load(f"{a.artifacts}/scaler.joblib")
    Xs = scaler.transform(X)

    rf = joblib.load(f"{a.artifacts}/rf.joblib")
    rf_pred = rf.predict(Xs)

    mlp = MLP(n_in=Xs.shape[1])
    mlp.load_state_dict(torch.load(f"{a.artifacts}/mlp.pt", map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        mlp_pred = (torch.sigmoid(mlp(torch.tensor(Xs, dtype=torch.float32))) >= 0.5).numpy().astype(int)

    y = np.full(len(X), a.label)
    print(f"{a.pcap}: {len(X)} Deauth frames recaptured")
    if a.label == 1:
        print(f"  RF  recall (still caught as attack): {(rf_pred == y).mean():.3f}")
        print(f"  MLP recall (still caught as attack): {(mlp_pred == y).mean():.3f}")
    else:
        print(f"  RF  false-alarm rate: {(rf_pred != y).mean():.3f}")
        print(f"  MLP false-alarm rate: {(mlp_pred != y).mean():.3f}")


if __name__ == "__main__":
    main()
