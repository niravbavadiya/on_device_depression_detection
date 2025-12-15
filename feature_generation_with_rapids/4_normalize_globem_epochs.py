#!/usr/bin/env python3
"""
Normalize GLOBEM-style epoch features across ALL modality CSVs under $BASE/data_raw/<pid>
and write a single merged file per pid: $BASE/data_raw/<pid>/rapids.csv.

- Column naming preserved: 'fam:raw:epoch' -> 'fam:raw_norm:epoch'
  e.g., f_screen:..._locmap_exercise -> f_screen:..._locmap_exercise_norm:14dhist
- Normalization per pid (if 'pid' present), otherwise over the file as-is.
- Non-feature cols: only 'pid' and 'date' are treated as keys; everything else is a feature.
- By default we write ONLY *_norm columns + keys; --keep-raw to include originals too.

Usage examples:
  # Process one pid
  python normalize_globem_epochs.py --base /path/to/feature_generation_thesis --pid P001

  # Process all pids under $BASE/data_raw/*
  python normalize_globem_epochs.py --base /path/to/feature_generation_thesis
"""

import argparse, os, glob, re
import pandas as pd
import numpy as np

# --------- args ---------
def parse_args():
    p = argparse.ArgumentParser(description="Normalize GLOBEM-style epoch features and merge modalities to rapids.csv")
    p.add_argument("--base", required=True, help="Project root (contains data_raw/<pid>)")
    p.add_argument("--pid",  help="Single PID to process; if omitted, process all PIDs under data_raw/")
    p.add_argument("--keep-raw", action="store_true", help="Include original (non-normalized) columns alongside *_norm")
    return p.parse_args()

# --------- helpers ---------
def norm_path(p: str) -> str:
    p = os.path.expanduser(str(p)).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/[a-zA-Z]/", p):
        p = f"{p[1].upper()}:{p[2:]}"
    return os.path.normpath(p)

def is_feature_col(c: str) -> bool:
    # Keep only 'pid' and 'date' as keys; everything else is treated as feature
    return c not in ("pid","date")

def parse_globem_col(col: str):
    """Return (family, raw_name, epoch) for 'f_xxx:raw_name:epoch' columns; else (None, None, None)."""
    parts = col.split(":", 2)
    if len(parts) != 3:
        return None, None, None
    return parts[0], parts[1], parts[2]

def norm_colname(col: str) -> str:
    fam, raw, ep = parse_globem_col(col)
    if fam is None:
        # For any non 'fam:raw:epoch' features, just append _norm
        return col + "_norm"
    return f"{fam}:{raw}_norm:{ep}"

def normalize_df(df: pd.DataFrame, keep_raw: bool=False) -> pd.DataFrame:
    """
    Return a new DF with keys ('pid','date') plus *_norm columns for each feature.
    If keep_raw=True, original feature columns are also included.
    """
    out = df.copy()
    # coerce numeric
    feat_cols = [c for c in out.columns if is_feature_col(c) and "_norm:" not in c and not c.endswith("_norm")]
    for c in feat_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # group per pid if present
    if "pid" in out.columns:
        groups = [(pid, grp) for pid, grp in out.groupby("pid", dropna=False)]
        key_cols = ["pid","date"]
    else:
        groups = [(None, out)]
        key_cols = ["date"]

    frames = []
    for _, grp in groups:
        # ensure keys exist
        if "date" not in grp.columns:
            raise SystemExit("Input file missing required 'date' column")
        res = grp[key_cols].copy()

        for c in feat_cols:
            s = grp[c]
            if not pd.api.types.is_numeric_dtype(s):
                continue

            if np.all(np.isnan(s.values)):
                z = s.copy() * np.nan
            else:
                med = np.nanmedian(s.values)
                q05 = np.nanquantile(s.values, 0.05)
                q95 = np.nanquantile(s.values, 0.95)
                scale = q95 - q05
                if scale and np.isfinite(scale) and scale > 0:
                    z = (s - med) / scale
                else:
                    z = s.copy() * 0.0
                    z[pd.isna(s)] = np.nan

            res[norm_colname(c)] = z.values

        if keep_raw:
            res = res.merge(grp[key_cols + feat_cols], on=key_cols, how="left")

        frames.append(res)

    return pd.concat(frames, ignore_index=True)

def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return pd.DataFrame()

# --------- main pipeline ---------
def process_pid(base: str, pid: str, keep_raw: bool=False):
    pid_dir = os.path.join(base, "data_raw", pid)
    if not os.path.isdir(pid_dir):
        print(f"[warn] pid dir not found: {pid_dir}")
        return

    # Collect modality files (skip our final output and any hidden/system files)
    files = sorted(glob.glob(os.path.join(pid_dir, "*.csv")))
    files = [f for f in files if os.path.basename(f).lower() not in ("rapids.csv",)]
    if not files:
        print(f"[warn] no CSV files in {pid_dir}")
        return

    combined = None
    for f in files:
        df = read_csv_safe(f)
        if df.empty:
            print(f"[info] skip empty: {f}")
            continue

        # normalize this modality
        normed = normalize_df(df, keep_raw=keep_raw)

        # merge into combined on keys
        keys = ["date"] + (["pid"] if "pid" in normed.columns else [])
        if combined is None:
            combined = normed
        else:
            combined = combined.merge(normed, on=keys, how="outer")

        print(f"[ok] normalized & merged: {os.path.basename(f)}  -> cols now={combined.shape[1]}")

    if combined is None or combined.empty:
        print(f"[info] nothing to write for {pid}")
        return

    # sort rows
    if "pid" in combined.columns:
        combined = combined.sort_values(["pid","date"])
    else:
        combined = combined.sort_values(["date"])

    # write output
    out_path = os.path.join(pid_dir, "rapids.csv")
    bak = out_path + ".bak"
    if not os.path.exists(bak):
        try:
            combined.to_csv(bak, index=False)
            print(f"[info] backup written: {bak}")
        except Exception:
            pass

    combined.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}  rows={combined.shape[0]}  cols={combined.shape[1]}")

def main():
    args = parse_args()
    base = norm_path(args.base)

    if args.pid:
        process_pid(base, args.pid, keep_raw=args.keep_raw)
    else:
        root = os.path.join(base, "data_raw")
        pids = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        if not pids:
            print(f"[warn] no PID folders under {root}")
            return
        for pid in pids:
            process_pid(base, pid, keep_raw=args.keep_raw)

if __name__ == "__main__":
    main()
