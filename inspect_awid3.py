#!/usr/bin/env python3
"""
Quick look at an AWID3-CSV file (use ONE small split file, ~50,000 rows).

    python inspect_awid3.py path/to/one_file.csv
    python inspect_awid3.py path/to/one_file.csv --nrows 20000 --out columns_report.csv

Prints: shape, label-column guess and its values, column types, missing-value share,
and how many columns could be used as numeric features. Run this first, then check the
settings at the top of awid3_baseline.py (LABEL_COL, DROP_REGEX) against what you see.
"""
import argparse
import re

import pandas as pd

from awid3_common import find_label_col, read_awid3, to_numeric_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--nrows", type=int, default=50000)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--out", default=None, help="optional CSV of the per-column report")
    a = ap.parse_args()

    df = read_awid3(a.csv, nrows=a.nrows)
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    label = a.label_col or find_label_col(df)
    print(f"Label column guess: {label!r}")
    if label in df.columns:
        print(df[label].value_counts(dropna=False).head(20).to_string(), "\n")

    rows = []
    for c in df.columns:
        s = df[c]
        miss = s.isna().mean()
        num = to_numeric_series(s)
        numeric_share = num.notna().mean() if s.notna().any() else 0.0
        rows.append(dict(column=c, dtype=str(s.dtype), missing=round(miss, 3),
                         n_unique=s.nunique(dropna=True), numeric_share=round(numeric_share, 3),
                         example=str(s.dropna().iloc[0])[:40] if s.notna().any() else ""))
    rep = pd.DataFrame(rows)
    usable = rep[(rep.missing < 0.5) & (rep.numeric_share > 0.9) & (rep.n_unique > 1)]
    print(f"Columns that look usable as numeric features: {len(usable)} of {len(rep)}")
    layers = {"wlan/radiotap (MAC layer)": r"^(wlan|radiotap|wlan_radio)",
              "frame": r"^frame", "ip/tcp/udp": r"^(ip|tcp|udp|ipv6)",
              "app layer (tls/http/dns/...)": r"^(tls|ssl|http|dns|dhcp|ssh|eap|arp|icmp)"}
    for name, pat in layers.items():
        print(f"  {name}: {rep.column.str.contains(pat, flags=re.I, regex=True).sum()} columns")
    print("\nFirst 40 columns:")
    print(rep.head(40).to_string(index=False))
    if a.out:
        rep.to_csv(a.out, index=False)
        print(f"\nFull report written to {a.out}")


if __name__ == "__main__":
    main()
