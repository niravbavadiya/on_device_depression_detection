#!/usr/bin/env python3
"""
3_locmap_from_polygons.py  (with DEBUG)
---------------------------------
Add *_locmap_* features to RAPIDS phone_screen.csv for ONE weekly chunk.

Inputs (explicit full paths from shell):
  --pid <PID>
  --locations  <BASE/aware_raw_data/<pid>/<chunk>/locations.csv>
  --polygons   <BASE/aware_raw_data/<pid>/locmap/locmap.csv>   (per-PID; ignore pid column)
  --screen     <BASE/rapids_out/processed/features/<pid>/phone_screen.csv>
  --tz         (default Europe/Berlin)
  --max-gap-min  (default 15)  forward-fill GPS up to N minutes
  --buffer-m     (default 0)   optional polygon buffer (meters) to counter GPS jitter

What it does:
  1) Build a per-minute category timeline from locations.csv using polygons.
  2) For each row in phone_screen.csv, parse local_segment "label#start,end"
     and allocate additive base metrics across categories proportionally to the
     minute fractions in [start,end). Derive avg = sum/count per category.

DEBUG FLAGS:
  --debug 1
  --debug-sample N
  --debug-dump-polys PATH
  --debug-dump-minutes PATH

Leaves file unchanged if no polygon coverage intersects that week.
"""

import os, re, argparse, math, sys
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.geometry import Point

# Category order for tie-breaks (after priority): lower index wins
CATEGORIES = ["home","study","living","exercise","greens"]

# Base metrics we try to allocate (row-level, NOT segment-suffixed)
BASE_METRICS = [
    "phone_screen_rapids_countepisodeunlock",
    "phone_screen_rapids_sumdurationunlock",
    "phone_screen_rapids_avgdurationunlock",  # derived
    # keep others (max/min/std/firstuseafter00) empty for now
]

def norm_path(p: str) -> str:
    """Normalize /c/... -> C:/... on Windows, and collapse slashes."""
    p = os.path.expanduser(str(p)).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/[a-zA-Z]/", p):
        p = f"{p[1].upper()}:{p[2:]}"
    return os.path.normpath(p)

def parse_local_segment(s: str, tz: str):
    """
    s = "afternoon#2025-10-13 12:00:00,2025-10-13 17:00:00"
    return ("afternoon", start_local, end_local)
    """
    lab, window = s.split("#", 1)
    start_s, end_s = window.split(",", 1)
    start = pd.Timestamp(start_s.strip()).tz_localize(tz)
    end   = pd.Timestamp(end_s.strip()).tz_localize(tz)
    return lab.strip(), start, end

def approx_deg_buffer(lat_deg: float, meters: float) -> float:
    """Return an approximate degree buffer for a given latitude (used isotropically)."""
    if meters <= 0: return 0.0
    # 1 degree latitude ~ 111_320 m; lon scale varies with cos(lat)
    # We use a conservative isotropic buffer in degrees using latitude scale.
    return meters / 111_320.0

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _h(x):
        x = str(x)
        x = x.replace("\ufeff", "")  # strip BOM if present
        return re.sub(r"\s+", " ", x).strip().lower()
    df.rename(columns=_h, inplace=True)
    return df

def load_polygons(polygons_csv: str, buffer_m: float = 0.0, debug: bool=False, dump_path: str=""):
    """
    Load per-PID polygons; ignore any 'pid' values in file.
    Return a list of rows: [{category, priority, polygon}], sorted by (priority asc, category rank asc).
    """
    if not os.path.isfile(polygons_csv):
        print(f"[info] polygons csv missing: {polygons_csv}")
        return []

    try:
        raw = pd.read_csv(polygons_csv)
    except Exception as e:
        print(f"[error] reading polygons csv failed: {e}")
        return []

    if debug:
        print(f"[debug] polygons.csv columns (raw): {list(raw.columns)}")
        print(f"[debug] polygons.csv head (raw):")
        print(raw.head(5).to_string(index=False))

    df = _clean_headers(raw)

    if debug:
        print(f"[debug] polygons.csv columns (clean): {list(df.columns)}")

    need = {"category","priority","wkt"}
    if not need.issubset(df.columns):
        print(f"[warn] polygons missing required columns {sorted(need)}; found {df.columns.tolist()}")
        return []

    df = df.copy()
    # normalize category strings strictly (no mapping)
    df["category"] = df["category"].astype(str).str.replace("\u00A0"," ").str.strip().str.lower()
    if debug:
        print(f"[debug] unique categories (cleaned): {sorted(df['category'].unique().tolist())}")

    df = df[df["category"].isin(CATEGORIES)]
    if df.empty:
        if debug:
            print(f"[debug] after category filter {CATEGORIES}, 0 rows remain")
        return []

    # priority + geometry
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce")
    bad_pri = df["priority"].isna().sum()
    df["priority"] = df["priority"].fillna(100).astype(int)

    def _to_geom(w, i):
        try:
            return wkt.loads(w)
        except Exception as e:
            if debug:
                print(f"[debug] bad WKT at row index {i}: {str(e)[:120]}")
            return None

    df["polygon"]  = [ _to_geom(w, i) for i, w in enumerate(df["wkt"]) ]
    bad_wkt = df["polygon"].isna().sum()
    df = df[df["polygon"].notna()]
    if df.empty:
        if debug:
            print(f"[debug] all WKT failed to parse; check WKT format (lon lat pairs)")
        return []

    if buffer_m and buffer_m > 0:
        # Apply a crude degree buffer based on polygon centroid latitude
        geoms = []
        for g in df["polygon"]:
            lat = g.centroid.y
            ddeg = approx_deg_buffer(lat, buffer_m)
            geoms.append(g.buffer(ddeg))
        df["polygon"] = geoms

    if debug:
        print(f"[debug] polygons loaded stats: rows={len(df)}, bad_priority={bad_pri}, bad_wkt={bad_wkt}")
        print("[debug] polygons by category:")
        print(df["category"].value_counts().to_string())
        # show a couple polygon centroids for sanity
        try:
            cents = df["polygon"].apply(lambda g: (round(g.centroid.x,7), round(g.centroid.y,7)))
            print("[debug] sample polygon centroids (lon,lat):", cents.head(5).tolist())
        except Exception:
            pass

    cat_rank = {c:i for i,c in enumerate(CATEGORIES)}
    rows = df[["category","priority","polygon"]].to_dict("records")
    rows.sort(key=lambda r: (r["priority"], cat_rank.get(r["category"], 999)))

    if dump_path:
        try:
            out = df[["category","priority","wkt"]].copy()
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            out.to_csv(dump_path, index=False)
            if debug:
                print(f"[debug] dumped cleaned polygons to: {dump_path}")
        except Exception as e:
            print(f"[debug] failed dumping cleaned polygons: {e}")

    return rows

def build_minute_series(loc_path: str, polys: list, tz_name: str = "Europe/Berlin",
                        max_gap_min: int = 15, debug: bool=False, debug_sample: int=10, dump_path: str="") -> pd.Series:
    """
    From locations.csv build a per-minute category series (index=local minute, value=category).
    Uses 'covers' (boundary-inclusive) and forward-fills within gaps up to max_gap_min.
    """
    if not os.path.isfile(loc_path):
        print(f"[info] no locations file: {loc_path}")
        return pd.Series(dtype=object)

    try:
        loc = pd.read_csv(loc_path)
    except Exception as e:
        print(f"[error] reading locations csv failed: {e}")
        return pd.Series(dtype=object)

    if debug:
        print(f"[debug] locations.csv columns: {list(loc.columns)}")
        print("[debug] locations.csv head:")
        print(loc.head(5).to_string(index=False))

    need = {"timestamp","double_latitude","double_longitude"}
    if not need.issubset(loc.columns):
        print(f"[warn] locations missing required columns {sorted(need)}; skipping.")
        return pd.Series(dtype=object)

    if not len(polys):
        if debug:
            print("[debug] polygons list is empty before classification")
        return pd.Series(dtype=object)

    # Time handling: AWARE timestamps are epoch ms; convert to UTC then to local tz
    loc = loc.copy()
    loc["ts_utc"] = pd.to_datetime(pd.to_numeric(loc["timestamp"], errors="coerce"), unit="ms", utc=True)
    loc = loc.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    if loc.empty:
        if debug:
            print("[debug] after ts_utc parse, locations are empty")
        return pd.Series(dtype=object)

    if debug:
        print("[debug] locations sample (lon,lat,ts_utc):")
        for i,row in enumerate(loc.head(debug_sample).itertuples(index=False), start=1):
            print(f"   #{i:02d} {getattr(row,'double_longitude',None)} {getattr(row,'double_latitude',None)} {getattr(row,'timestamp',None)} -> {getattr(row,'ts_utc',None)}")

    # Map each point -> best category by priority (covers includes boundary)
    pts  = [Point(lon, lat) for lon,lat in zip(loc["double_longitude"], loc["double_latitude"])]
    cats = []
    for k, p in enumerate(pts):
        win = None
        for row in polys:  # sorted by (priority asc, cat_rank asc)
            try:
                hit = row["polygon"].covers(p)
            except Exception:
                hit = False
            if hit:
                win = row["category"]; break
        cats.append(win)

    loc["cat"] = cats
    if debug:
        uniq = pd.Series(cats).value_counts(dropna=False).to_dict()
        print(f"[debug] classification results counts (including None): {uniq}")
        print(f"[debug] first {min(debug_sample, len(cats))} classifications:", cats[:min(debug_sample, len(cats))])

    loc = loc.dropna(subset=["cat"])
    if loc.empty:
        if debug:
            print("[debug] after dropping None categories, 0 rows remain (no point fell in any polygon)")
        return pd.Series(dtype=object)

    # Minute timeline
    loc["ts_local"]  = loc["ts_utc"].dt.tz_convert(tz_name)
    loc["min_local"] = loc["ts_local"].dt.floor("T")
    dfm = loc.sort_values("ts_local").groupby("min_local", as_index=True).tail(1)
    s = dfm.set_index("min_local")["cat"].sort_index()

    # Forward-fill within blocks separated by > max_gap_min
    gap = pd.Timedelta(minutes=max_gap_min)
    blocks = (s.index.to_series().diff().gt(gap)).cumsum()
    pieces = []
    for _, sb in s.groupby(blocks):
        full_idx = pd.date_range(sb.index.min(), sb.index.max(), freq="T", tz=sb.index.tz)
        pieces.append(sb.reindex(full_idx).ffill())
    out = pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=object)

    if debug:
        print(f"[debug] per-minute series length: {len(out)}  (unique cats: {out.value_counts().to_dict() if not out.empty else {}})")
        if dump_path:
            try:
                df_dump = out.rename("cat").reset_index().rename(columns={"min_local":"minute"})
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                df_dump.to_csv(dump_path, index=False)
                print(f"[debug] dumped minute series to: {dump_path}")
            except Exception as e:
                print(f"[debug] failed dumping minute series: {e}")

    return out

def list_present_base_metrics(screen_df: pd.DataFrame) -> list:
    present = []
    for m in BASE_METRICS:
        if m in screen_df.columns:
            present.append(m)
    return present

def ensure_locmap_columns(scr: pd.DataFrame, metrics: list) -> None:
    for m in metrics:
        for cat in CATEGORIES:
            col = f"{m}_locmap_{cat}"
            if col not in scr.columns:
                scr[col] = pd.NA

def allocate_additive(value, frac):
    if pd.isna(value) or value == 0 or frac == 0:
        return 0.0
    return float(value) * float(frac)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=False, default="(single-file)")
    ap.add_argument("--locations",  required=True, help="FULL path to weekly locations.csv")
    ap.add_argument("--polygons",   required=True, help="FULL path to per-PID locmap.csv (ignore pid column)")
    ap.add_argument("--write_csv",     required=True, help="FULL path to phone_screen.csv to update")
    ap.add_argument("--tz",         default="Europe/Berlin")
    ap.add_argument("--max-gap-min", type=int, default=15)
    ap.add_argument("--buffer-m",    type=float, default=0.0)
    # debug
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--debug-sample", type=int, default=10)
    ap.add_argument("--debug-dump-polys", type=str, default="")
    ap.add_argument("--debug-dump-minutes", type=str, default="")
    args = ap.parse_args()

    debug = bool(args.debug)
    pid        = str(args.pid)
    loc_path   = norm_path(args.locations)
    poly_csv   = norm_path(args.polygons)
    write_csv = norm_path(args.write_csv)

    print(f"[paths] locations={loc_path}")
    print(f"[paths] polygons ={poly_csv}")
    print(f"[paths] screen   ={write_csv}")

    # Load polygons (per-PID file; ignore pid values)
    if not os.path.isfile(poly_csv):
        print("[info] polygons file missing/empty; leaving screen unchanged.")
        return
    try:
        # show raw columns immediately
        _rows_preview = pd.read_csv(poly_csv, nrows=5)
        if debug:
            print(f"[debug] polygons raw header = {list(_rows_preview.columns)}")
            print(f"[debug] polygons preview:\n{_rows_preview.to_string(index=False)}")
    except Exception as e:
        print(f"[error] failed opening polygons file: {e}")
        return

    polygons = load_polygons(poly_csv, buffer_m=float(args.buffer_m),
                             debug=debug, dump_path=args.debug_dump_polys)
    if not polygons:
        print("[info] no usable polygons; leaving screen unchanged.")
        return
    else:
        if debug:
            cats = [r["category"] for r in polygons]
            print(f"[debug] usable polygons loaded: {len(polygons)} (by priority/category order). Category counts: "
                  f"{pd.Series(cats).value_counts().to_dict()}")

    # Build minute series for this week
    smin = build_minute_series(
        loc_path, polygons,
        tz_name=args.tz,
        max_gap_min=int(args.max_gap_min),
        debug=debug,
        debug_sample=int(args.debug_sample),
        dump_path=args.debug_dump_minutes
    )
    if smin.empty:
        print("[info] No in-polygon GPS minutes for this pid/week; screen file unchanged.")
        return

    # Load phone_screen and verify required columns
    if not os.path.isfile(write_csv):
        print(f"[warn] phone_screen.csv not found: {write_csv}")
        return
    try:
        scr = pd.read_csv(write_csv)
    except Exception as e:
        print(f"[error] failed reading phone_screen.csv: {e}")
        return

    if "local_segment" not in scr.columns:
        raise SystemExit("[error] phone_screen.csv must have 'local_segment' like 'afternoon#YYYY-mm-dd HH:MM:SS,YYYY-mm-dd HH:MM:SS'.")
    
    # File is per-participant; process all rows
    idxs = scr.index
    if len(idxs) == 0:
        print("[info] phone_screen.csv has 0 rows; nothing to update.")
        return

    # Which base metrics exist?
    base_metrics = list_present_base_metrics(scr)
    if not base_metrics:
        print("[info] No base screen metrics present; nothing to allocate.")
        return

    # Ensure *_locmap_* columns exist (no segment suffix; row is the window)
    ensure_locmap_columns(scr, base_metrics)

    #if debug:
    #    print(f"[debug] processing screen rows for pid={pid}: count={len(idxs)}")
    #    show_cols = [pid_col,"local_segment"] + base_metrics
    #    print("[debug] phone_screen sample rows:")
    #    print(scr.loc[idxs].head(3)[show_cols].to_string(index=False))

    # Allocate per row using its local_segment window
    tz = args.tz
    updates = 0
    for idx in idxs:
        seg = scr.at[idx, "local_segment"]
        if pd.isna(seg):
            continue
        try:
            lbl, t0, t1 = parse_local_segment(str(seg), tz)
        except Exception as e:
            if debug:
                print(f"[debug] bad local_segment at row {idx}: {seg!r} -> {e}")
            continue

        sub = smin.loc[(smin.index >= t0) & (smin.index < t1)]
        if sub.empty:
            if debug:
                print(f"[debug] row {idx} seg={lbl} {t0}..{t1} -> 0 minutes matched")
            continue

        counts = sub.value_counts()
        total  = counts.sum()
        fracs  = {c: (counts.get(c, 0) / total) for c in CATEGORIES}

        if debug:
            print(f"[debug] row {idx} seg={lbl} window={t0}..{t1} minutes={int(total)} counts={counts.to_dict()} fracs={fracs}")

        # Additive metrics
        if "phone_screen_rapids_countepisodeunlock" in base_metrics:
            base_val = scr.at[idx, "phone_screen_rapids_countepisodeunlock"]
            if not pd.isna(base_val):
                for cat in CATEGORIES:
                    scr.at[idx, f"phone_screen_rapids_countepisodeunlock_locmap_{cat}"] = allocate_additive(base_val, fracs[cat])

        if "phone_screen_rapids_sumdurationunlock" in base_metrics:
            base_val = scr.at[idx, "phone_screen_rapids_sumdurationunlock"]
            if not pd.isna(base_val):
                for cat in CATEGORIES:
                    scr.at[idx, f"phone_screen_rapids_sumdurationunlock_locmap_{cat}"] = allocate_additive(base_val, fracs[cat])

        updates += 1

    # Derive avg = sum / count (row-wise)
    have = set(base_metrics)
    if {"phone_screen_rapids_avgdurationunlock",
        "phone_screen_rapids_sumdurationunlock",
        "phone_screen_rapids_countepisodeunlock"}.issubset(have) and updates > 0:
        for cat in CATEGORIES:
            sum_col = f"phone_screen_rapids_sumdurationunlock_locmap_{cat}"
            cnt_col = f"phone_screen_rapids_countepisodeunlock_locmap_{cat}"
            avg_col = f"phone_screen_rapids_avgdurationunlock_locmap_{cat}"
            # safe division
            sums = pd.to_numeric(scr.loc[idxs, sum_col], errors="coerce")
            cnts = pd.to_numeric(scr.loc[idxs, cnt_col], errors="coerce")
            vals = []
            for a, b in zip(sums, cnts):
                if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
                    vals.append(np.nan)
                else:
                    vals.append(float(a)/float(b))
            scr.loc[idxs, avg_col] = vals

    if updates == 0:
        print("[info] No screen rows got allocations (likely no GPS minutes in any segment).")
        return

    # Backup (once) then write in-place
    bak = write_csv + ".bak"
    if not os.path.exists(bak):
        try:
            scr.to_csv(bak, index=False)
            print(f"[info] backup written: {bak}")
        except Exception:
            pass

    scr.to_csv(write_csv, index=False)
    print(f"[ok] updated: {write_csv}")

if __name__ == "__main__":
    main()
