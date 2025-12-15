#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, glob
import pandas as pd
import numpy as np

"""
Batch: build epoch features + 7/14-day histories from RAPIDS all_sensor_features.csv,
then write per-modality CSVs under feature_generation_thesis/data_raw/<pid>/

Modalities (by 'family' prefix of features):
  f_blue   -> bluetooth.csv
  f_call   -> calls.csv
  f_screen -> screen.csv
  f_wifi   -> wifi.csv
  f_loc    -> location.csv
  f_steps  -> steps.csv
  f_slp    -> sleep.csv
  f_misc   -> misc.csv

Both epoch-level and 7/14-day history features are included in each modality CSV.
If a CSV for a PID+modality already exists, values are union-merged with preference
for newly computed (non-null) cells.
"""

# --------------------- helpers ---------------------

def family_prefix(col: str) -> str:
    c = col.lower()
    if c.startswith("phone_bluetooth"):        return "f_blue"
    if c.startswith("phone_calls"):            return "f_call"
    if c.startswith("phone_screen"):           return "f_screen"
    if c.startswith("phone_wifi_visible"):     return "f_wifi"
    if c.startswith("phone_wifi_connected"):   return "f_wifi"
    if c.startswith("phone_locations"):        return "f_loc"
    if c.startswith("fitbit_steps"):           return "f_steps"
    if c.startswith("fitbit_sleep"):           return "f_slp"
    return "f_misc"

SUM_TOKENS  = ("sum", "count", "uniquedevices", "duration", "totaldistance")
MEAN_TOKENS = ("avg", "mean", "std", "entropy", "efficiency", "ratio", "first", "last", "var", "max", "min")

def agg_kind(col: str) -> str:
    name = col.lower()
    if any(t in name for t in MEAN_TOKENS): return "mean"
    if any(t in name for t in SUM_TOKENS):  return "sum"
    return "sum"

def first_non_null(series: pd.Series):
    for v in series:
        if pd.notnull(v):
            return v
    return np.nan

def rename_epoch_col(raw_feat: str, epoch: str) -> str:
    return f"{family_prefix(raw_feat)}:{raw_feat}:{epoch}"

def rename_hist_col(raw_feat: str, suffix: str) -> str:
    return f"{family_prefix(raw_feat)}:{raw_feat}:{suffix}"

def _norm_date(s: pd.Series) -> pd.Series:
    # ensure pure date (UTC-normalized calendar day) for stable joins
    return pd.to_datetime(s, errors="coerce").dt.normalize()

# ----------------- epochs pivot --------------------

def build_epochs_for_pid(csv_path: str, pid: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    need = {"local_segment_label","local_segment_start_datetime"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[{pid}] Missing required column(s): {miss} in {csv_path}")

    df = df.copy()
    df["date"] = _norm_date(df["local_segment_start_datetime"])
    df["label"] = df["local_segment_label"].astype(str).str.lower().str.strip()

    feat_cols = [c for c in df.columns if c.startswith(("phone_","fitbit_"))]
    if not feat_cols:
        raise SystemExit(f"[{pid}] No feature columns found in {csv_path}")

    # One row per (date,label): take first non-null per feature
    df = df.sort_values(["date","label","local_segment_start_datetime"])
    agg = (df.groupby(["date","label"], sort=True, dropna=False)[feat_cols]
             .agg(first_non_null)
             .reset_index())

    # Pivot labels to columns -> "feature:epoch"
    wide = agg.pivot(index="date", columns="label", values=feat_cols)

    # Flatten MultiIndex → dataframe with columns "feature:epoch"
    flat = pd.DataFrame(index=wide.index)
    labels = list(wide.columns.levels[1]) if isinstance(wide.columns, pd.MultiIndex) else []
    for feat in feat_cols:
        for lbl in labels:
            if (feat, lbl) in wide.columns:
                flat[f"{feat}:{lbl}"] = wide[(feat, lbl)]

    flat = flat.reset_index()
    flat.insert(0, "pid", pid)

    # Rename to GLOBEM style (f_xxx:raw:epoch)
    new_cols = []
    for c in flat.columns:
        if c in ("pid","date"):
            new_cols.append(c)
        elif ":" in c:
            raw, ep = c.split(":", 1)
            new_cols.append(rename_epoch_col(raw, ep))
        else:
            new_cols.append(c)
    flat.columns = new_cols
    return flat

# --------------- 7/14-day histories ----------------

def pick_one_daily_row(df: pd.DataFrame) -> pd.DataFrame:
    seg = df["local_segment_label"].astype(str).str.lower()
    df = df.copy()
    df["date"] = _norm_date(df["local_segment_start_datetime"])

    day = df[seg.isin(["allday","daily"])].copy()
    if not day.empty:
        day = day.sort_values(["date","local_segment_start_datetime"])
        day = day.drop_duplicates(subset=["date"], keep="first")
        return day

    # Fallback: first non-null per column across that date
    feat_cols = [c for c in df.columns if c.startswith(("phone_","fitbit_"))]
    df = df.sort_values(["date","local_segment_start_datetime"])
    collapsed = (df.groupby("date", sort=True, dropna=False)[feat_cols]
                   .agg(first_non_null)
                   .reset_index())
    collapsed["local_segment_label"] = "synthetic_allday"
    collapsed["local_segment_start_datetime"] = collapsed["date"].astype(str) + " 00:00:00"
    return collapsed

def build_hist_for_pid(csv_path: str, pid: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ("local_segment_label","local_segment_start_datetime"):
        if col not in df.columns:
            raise SystemExit(f"[{pid}] Missing required column: {col} in {csv_path}")

    daily = pick_one_daily_row(df)
    daily["date"] = pd.to_datetime(daily["date"])

    feat_cols = [c for c in daily.columns if c.startswith(("phone_","fitbit_"))]
    daily = daily[["date"] + feat_cols].set_index("date").asfreq("D")

    out = pd.DataFrame(index=daily.index)
    for c in feat_cols:
        s = daily[c]
        if agg_kind(c) == "sum":
            base = s.fillna(0)
            out[rename_hist_col(c, "7dhist")]  = base.rolling(7,  min_periods=1).sum()
            out[rename_hist_col(c, "14dhist")] = base.rolling(14, min_periods=1).sum()
        else:
            base = s
            out[rename_hist_col(c, "7dhist")]  = base.rolling(7,  min_periods=1).mean()
            out[rename_hist_col(c, "14dhist")] = base.rolling(14, min_periods=1).mean()

    out = out.reset_index()
    out.insert(0, "pid", pid)
    return out

# --------------- merge/append logic ----------------

def resolve_columns_order(existing_cols, new_cols):
    """
    Keep existing column order; append any new columns at the end (sorted for stability).
    Always ensure ['pid','date'] are first.
    """
    ex = [c for c in existing_cols if c not in ("pid","date")]
    nw = [c for c in new_cols if c not in ("pid","date")]
    extra = [c for c in nw if c not in ex]
    ordered = ["pid", "date"] + ex + sorted(extra)
    return ordered

def union_merge_keep_new(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Union by (pid,date). For overlapping cells: prefer NEW values if present,
    otherwise keep EXISTING. Also carries columns present only in one side.
    """
    # Normalize key types
    existing = existing.copy()
    new = new.copy()
    existing["date"] = _norm_date(existing["date"])
    new["date"] = _norm_date(new["date"])
    existing["pid"] = existing["pid"].astype(str)
    new["pid"] = new["pid"].astype(str)

    # Index on keys
    e_idx = existing.set_index(["pid","date"])
    n_idx = new.set_index(["pid","date"])

    # Prefer new where it has non-null; otherwise use old
    resolved = n_idx.combine_first(e_idx)

    # Column order: keep existing order, append new ones
    ordered_cols = resolve_columns_order(existing.columns.tolist(), resolved.reset_index().columns.tolist())
    resolved = resolved.reset_index()
    # Add any missing ordered columns (if any) before reindex
    for c in ordered_cols:
        if c not in resolved.columns:
            resolved[c] = np.nan
    resolved = resolved.reindex(columns=ordered_cols)

    # Sort by pid/date for stability
    resolved = resolved.sort_values(["pid","date"]).reset_index(drop=True)
    return resolved

# --------------- modality splitting ----------------

MODALITY_FILES = {
    "f_blue":  "bluetooth.csv",
    "f_call":  "calls.csv",
    "f_screen":"screen.csv",
    "f_wifi":  "wifi.csv",
    "f_loc":   "location.csv",
    "f_steps": "steps.csv",
    "f_slp":   "sleep.csv",
    "f_misc":  "misc.csv",
}

def split_by_modality(df: pd.DataFrame) -> dict:
    """
    Return a dict: modality -> dataframe with ['pid','date', modality columns...]
    """
    out = {}
    cols = df.columns.tolist()
    for fam, _ in MODALITY_FILES.items():
        fam_cols = [c for c in cols if c.startswith(fam + ":")]
        if not fam_cols:
            continue
        sub = df[["pid","date"] + fam_cols].copy()
        out[fam] = sub
    return out

# ------------------ batch runner -------------------

def main():
    ap = argparse.ArgumentParser(description="Create per-modality epoch + 7/14-day history features per participant and save under feature_generation_thesis/data_raw/<pid>/")
    ap.add_argument("--base", default=".", help="Project root (or chunk root). Expects ./rapids_out/processed/features/*/all_sensor_features.csv")
    ap.add_argument("--features-subdir", default=os.path.join("rapids_out","processed","features"),
                    help="Relative path to features folder (default: rapids_out/processed/features)")
    ap.add_argument("--out-root", default=os.path.join("data_raw"),
                    help="Relative output root (default: feature_generation_thesis/data_raw)")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    feats_root = os.path.join(base, args.features_subdir)
    out_root  = os.path.join(base, args.out_root)

    pattern = os.path.join(feats_root, "*", "all_sensor_features.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[warn] No files matched {pattern}")
        sys.exit(0)

    os.makedirs(out_root, exist_ok=True)

    print(f"[info] Found {len(files)} participant file(s).")
    for csv_path in files:
        pid = os.path.basename(os.path.dirname(csv_path))
        out_pid_dir = os.path.join(out_root, pid)
        os.makedirs(out_pid_dir, exist_ok=True)

        try:
            # Build this-chunk dataframe (epochs + hist)
            epochs_wide = build_epochs_for_pid(csv_path, pid)
            hist_wide   = build_hist_for_pid(csv_path, pid)
            new_df      = pd.merge(epochs_wide, hist_wide, on=["pid","date"], how="outer", sort=True)

            # Split by modality and write/merge per file
            parts = split_by_modality(new_df)
            if not parts:
                print(f"[skip] {pid}: no modality columns found.")
                continue

            for fam, subdf in parts.items():
                fname = MODALITY_FILES.get(fam, f"{fam}.csv")
                fpath = os.path.join(out_pid_dir, fname)

                if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                    existing_df = pd.read_csv(fpath, low_memory=False)
                    # Ensure required keys exist even if older file had different schema
                    if "pid" not in existing_df.columns or "date" not in existing_df.columns:
                        raise SystemExit(f"[{pid}] Existing file at {fpath} lacks 'pid' or 'date' columns.")
                    merged = union_merge_keep_new(existing_df, subdf)
                    merged.to_csv(fpath, index=False)
                    print(f"[ok] {pid}:{fam} -> merged {fname}  rows={merged.shape[0]} cols={merged.shape[1]}")
                else:
                    subdf.to_csv(fpath, index=False)
                    print(f"[ok] {pid}:{fam} -> wrote {fname}  rows={subdf.shape[0]} cols={subdf.shape[1]}")

        except Exception as e:
            print(f"[error] {pid}: {e}")

if __name__ == "__main__":
    # Enable CoW only if this pandas has the option (older pandas won't)
    try:
        if hasattr(pd.options, "mode") and hasattr(pd.options.mode, "copy_on_write"):
            pd.options.mode.copy_on_write = True
    except Exception:
        pass
    main()

