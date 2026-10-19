#!/usr/bin/env python3
"""
List every CSV piece with its label counts, so you can see which pieces contain attack rows.

    python scan_pieces.py data/ --out piece_labels.csv

Reads only the Label column of each piece (fast, low memory). Prints a summary per attack group.
"""
import argparse
import os
import re

import pandas as pd

from awid3_common import binary_target, expand_paths, piece_label_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="piece_labels.csv")
    a = ap.parse_args()
    rows = []
    files = expand_paths(a.data)
    for i, f in enumerate(files, 1):
        counts = piece_label_counts(f)
        total = sum(counts.values())
        attack = sum(v for k, v in counts.items()
                     if binary_target(pd.Series([k])).iloc[0] == 1)
        grp = re.sub(r"[_\\-\\s]*\\d+$", "", os.path.splitext(os.path.basename(f))[0])
        rows.append(dict(file=os.path.basename(f), group=grp, rows=total, attack_rows=attack,
                         normal_rows=total - attack, path=f))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {attack:,} attack / {total:,}")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("\nPer group:")
    g = df.groupby("group").agg(pieces=("file", "count"), rows=("rows", "sum"),
                                attack_rows=("attack_rows", "sum"),
                                pieces_with_attack=("attack_rows", lambda s: int((s > 0).sum())))
    print(g.to_string())
    print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
