"""Shared helpers for reading and cleaning AWID3-CSV files.

Notes from the AWID3-CSV README:
  * features span the MAC layer and the application layer
  * multi-valued cells had their commas replaced by "-"; empty cells are "-"
  * some TLS rows are only partially decrypted -> drop them (see drop_bad_tls_rows)
  * each source CSV was split into 50,000-row pieces, so you can load a handful of pieces
"""
import glob
import os
import re

import numpy as np
import pandas as pd

LABEL_CANDIDATES = ["label", "class", "attack", "target", "category"]
BENIGN_WORDS = {"normal", "benign", "0", "legitimate", "legit"}


def expand_paths(paths):
    """Accept files, folders, or glob patterns; return a sorted list of .csv files."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        elif any(ch in p for ch in "*?["):
            out += glob.glob(p, recursive=True)
        else:
            out.append(p)
    out = sorted(set(out))
    if not out:
        raise SystemExit("No CSV files found for the given path(s). Check DATA_PATHS.")
    return out


def read_awid3(path, nrows=None, usecols=None):
    """Read one CSV tolerantly (wide file, mixed types, a few overlong cells)."""
    return pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False,
                       na_values=["-", "", " ", "NA", "nan"], on_bad_lines="skip",
                       encoding_errors="replace")


def find_label_col(df):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in lower:
            return lower[cand]
    for c in df.columns:  # contains 'label'
        if "label" in c.lower():
            return c
    return df.columns[-1]  # README: the label is the extra (last) column


def _hex_to_float(x):
    try:
        return float(int(x, 16))
    except (TypeError, ValueError):
        return np.nan


def to_numeric_series(s):
    """Numeric conversion that also understands 0x... hex strings and True/False."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if s.dtype == bool:
        return s.astype(float)
    t = pd.to_numeric(s, errors="coerce")
    nn = s.notna() & t.isna()
    if nn.any():
        as_str = s[nn].astype(str).str.strip()
        hexmask = as_str.str.match(r"^0x[0-9a-fA-F]+$")
        if hexmask.any():
            t.loc[as_str[hexmask].index] = as_str[hexmask].map(_hex_to_float)
        low = as_str.str.lower()
        t.loc[low[low == "true"].index] = 1.0
        t.loc[low[low == "false"].index] = 0.0
    return t


def drop_bad_tls_rows(df):
    """README: TLS is not fully decrypted for every packet; remove such rows if flagged."""
    cols = [c for c in df.columns if re.search(r"tls.*(decrypt|malformed|expert)", c, re.I)]
    for c in cols:
        df = df[df[c].isna()]
    return df


def binary_target(series):
    """Return y in {0,1}: 0 = benign/normal, 1 = attack. Works for text or numeric labels."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) != 0).astype(int)
    txt = series.astype(str).str.strip().str.lower()
    return (~txt.isin(BENIGN_WORDS)).astype(int)
