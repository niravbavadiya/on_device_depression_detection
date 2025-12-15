#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patch_config.py
- Seeds/updates /rapids/data/config.yaml for RAPIDS.
- Participants mode: creates participant YAMLs based on your CSV.
- Features mode: enables selected sensors/providers, time segments,
  Bluetooth DORYAB, Locations (Barnett+Doryab), and builds + wires a locmap.
- Locmap: built from /rapids/data/external/aware_csv/fused_geofences.csv; if empty,
  a fallback locmap is created so _locmap_ features are emitted.
"""

import argparse, os, sys, subprocess, glob, re, json, math, csv

CFG = "/rapids/data/config.yaml"
VENDOR = "/rapids/data/vendor/python"
PARTICIPANT_CSV_DEFAULT = "/rapids/data/external/participant_files/participant_files.csv"

# ---------------- ruamel vendoring ----------------
def ensure_ruamel():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    try:
        from ruamel.yaml import YAML  # noqa
        return
    except Exception:
        os.makedirs(VENDOR, exist_ok=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-t", VENDOR, "ruamel.yaml"])
        if VENDOR not in sys.path:
            sys.path.insert(0, VENDOR)

def get_yaml():
    ensure_ruamel()
    from ruamel.yaml import YAML
    y = YAML()
    y.preserve_quotes = True
    return y

def flow_seq(items):
    from ruamel.yaml.comments import CommentedSeq
    s = CommentedSeq(items)
    try:
        s.yaml_set_flow_style()
    except Exception:
        try:
            s.fa.set_flow_style()
        except Exception:
            pass
    return s

# Path inside the container where participant_files.csv lives
PARTICIPANT_CSV_DEFAULT = "/rapids/data/external/participant_files/participant_files.csv"


def load_pids_from_participant_csv(csv_path: str):
    """
    Read all unique, non-empty PIDs from participant_files.csv.
    Falls back to ['P001'] if file/column is missing or empty.
    """
    if not os.path.exists(csv_path):
        print(f"[warn] participant_files.csv not found at {csv_path}; defaulting to P001", file=sys.stderr)
        return ["P001"]

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"[warn] participant_files.csv has no header; defaulting to P001", file=sys.stderr)
                return ["P001"]

            # Find the 'pid' column (case-insensitive)
            pid_field = None
            for name in reader.fieldnames:
                if name and name.strip().lower() == "pid":
                    pid_field = name
                    break

            if not pid_field:
                print(f"[warn] participant_files.csv has no 'pid' column; defaulting to P001", file=sys.stderr)
                return ["P001"]

            pids_raw = []
            for row in reader:
                val = (row.get(pid_field) or "").strip()
                if val:
                    pids_raw.append(val)

        # make unique but preserve order
        seen = set()
        pids = []
        for p in pids_raw:
            if p not in seen:
                seen.add(p)
                pids.append(p)

        if not pids:
            print(f"[warn] No non-empty PIDs found in {csv_path}; defaulting to P001", file=sys.stderr)
            return ["P001"]

        print(f"[info] Loaded {len(pids)} PID(s) from {csv_path}: {', '.join(pids)}")
        return pids

    except Exception as e:
        print(f"[warn] Failed to read participant_files.csv at {csv_path}: {e}; defaulting to P001", file=sys.stderr)
        return ["P001"]


def ensure_map(m, k):
    from ruamel.yaml.comments import CommentedMap
    if not isinstance(m, dict) or m is None:
        m = CommentedMap()
    if k not in m or not isinstance(m[k], dict):
        m[k] = CommentedMap()
    return m[k]

def set_inline_list(m, key, items):
    m[key] = flow_seq(items)

# ---------------- helpers for providers/blocks ----------------
def _get_sensor_block(cfg, sensor_key):
    blk = cfg.get(sensor_key)
    return blk if isinstance(blk, dict) else None

def set_provider(cfg, sensor_key, provider, **kv):
    from ruamel.yaml.comments import CommentedMap
    blk = _get_sensor_block(cfg, sensor_key)
    if not blk:
        return
    provs = blk.get("PROVIDERS")
    if not isinstance(provs, dict):
        provs = blk["PROVIDERS"] = CommentedMap()
    if provider not in provs or not isinstance(provs[provider], dict):
        provs[provider] = CommentedMap()
    for k, v in kv.items():
        provs[provider][k] = v

def set_src(cfg, sensor_key, provider, path_abs):
    set_provider(cfg, sensor_key, provider, SRC_SCRIPT=path_abs)

def disable_all(cfg, sensor_key):
    blk = _get_sensor_block(cfg, sensor_key)
    if not blk:
        return
    provs = blk.get("PROVIDERS")
    if not isinstance(provs, dict):
        return
    for v in provs.values():
        if isinstance(v, dict):
            v["COMPUTE"] = False

def set_container(cfg, sensor_key, filename):
    blk = _get_sensor_block(cfg, sensor_key)
    if blk is not None:
        blk["CONTAINER"] = filename

def set_compute_block(cfg, sensor_key, prefer=None, only=None):
    blk = cfg.get(sensor_key)
    if not isinstance(blk, dict):
        return
    provs = blk.get("PROVIDERS")
    if not isinstance(provs, dict):
        return
    if only:
        for n, v in provs.items():
            if isinstance(v, dict):
                v["COMPUTE"] = (n in only)
        return
    if prefer and prefer in provs and isinstance(provs[prefer], dict):
        for n, v in provs.items():
            if isinstance(v, dict):
                v["COMPUTE"] = (n == prefer)
    else:
        for n, v in provs.items():
            if isinstance(v, dict):
                v["COMPUTE"] = True

def apply_whitelist(cfg, allowed):
    """Final pass: set COMPUTE True only for allowed providers per sensor key."""
    for key, blk in list(cfg.items()):
        if not (isinstance(key, str) and key.startswith("PHONE_")):
            continue
        if key in ("PHONE_DATA_STREAMS", "PHONE_DATA_YIELD"):
            continue
        if not isinstance(blk, dict):
            continue
        provs = blk.get("PROVIDERS")
        if not isinstance(provs, dict):
            continue
        for pname, pblk in provs.items():
            if isinstance(pblk, dict):
                pblk["COMPUTE"] = (key in allowed and pname in allowed[key])

def _warn_if_missing(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("[warn] SRC_SCRIPT target(s) not found: " + ", ".join(missing), file=sys.stderr)

# ---------------- refine SRC_SCRIPT from repo tree ----------------
def find_provider_script(sensor_key, provider_key):
    s = sensor_key.lower()
    p = provider_key.lower()
    root = f"/rapids/src/features/{s}/{p}"
    if not os.path.isdir(root):
        return None
    for patt in ("*rapids.R", "*rapids.py", "*.R", "*.py"):
        cand = sorted(glob.glob(os.path.join(root, patt)))
        if cand:
            return cand[0]
    return None

def refine_src_scripts_from_tree(cfg, collect_changes=False):
    changes = []
    for k, blk in list(cfg.items()):
        if not (isinstance(k, str) and k.startswith("PHONE_")):
            continue
        if k in ("PHONE_DATA_STREAMS","PHONE_DATA_YIELD"):
            continue
        if not isinstance(blk, dict):
            continue
        provs = blk.get("PROVIDERS")
        if not isinstance(provs, dict):
            continue
        for prov_key, prov in provs.items():
            if not isinstance(prov, dict):
                continue
            script = find_provider_script(k, prov_key)
            if script and prov.get("SRC_SCRIPT") != script:
                prov["SRC_SCRIPT"] = script
                if collect_changes:
                    changes.append((k, prov_key, script))
    # Calls: ensure episodes
    if "PHONE_CALLS" in cfg:
        provs = cfg["PHONE_CALLS"].get("PROVIDERS", {})
        rapids = provs.get("RAPIDS")
        if isinstance(rapids, dict) and "FEATURES_TYPE" not in rapids:
            rapids["FEATURES_TYPE"] = "EPISODES"
            if collect_changes:
                changes.append(("PHONE_CALLS","RAPIDS","FEATURES_TYPE=EPISODES"))
    return changes

# ---------------- Locations: Barnett & Doryab helpers ----------------
def find_barnett_features_script():
    base = "/rapids/src/features/phone_locations/barnett"
    if not os.path.isdir(base):
        return None
    pat = re.compile(r"barnett_features\s*<-\s*function", re.IGNORECASE)
    for rfile in sorted(glob.glob(os.path.join(base, "*.R"))):
        try:
            with open(rfile, "r", encoding="utf-8", errors="ignore") as fh:
                if pat.search(fh.read()):
                    return rfile
        except Exception:
            continue
    return None

def enforce_barnett_script(cfg):
    path = find_barnett_features_script()
    if path:
        set_provider(cfg, "PHONE_LOCATIONS", "BARNETT", SRC_SCRIPT=path, COMPUTE=True)
        print(f"[info] BARNETT SRC_SCRIPT -> {path}")
    else:
        print("[warn] Could not find a file defining barnett_features under barnett/.")

def choose_doryab_script(prefer_main=True):
    candidates = [
        "/rapids/src/features/phone_locations/doryab/main.py",
        "/rapids/src/features/phone_locations/doryab/features.py",
        "/rapids/src/features/phone_locations/doryab/provider.py",
        "/rapids/src/features/phone_locations/doryab/compute_features.py",
        "/rapids/src/features/phone_locations/doryab/feature_extraction.py",
    ]
    if not prefer_main:
        candidates = candidates[1:] + candidates[:1]
    for c in candidates:
        if os.path.exists(c):
            return c
    any_py = [p for p in glob.glob("/rapids/src/features/phone_locations/doryab/*.py")
              if os.path.basename(p) != "add_doryab_extra_columns.py"]
    return any_py[0] if any_py else None

def ensure_doryab_provider(cfg, force_main_preference=True):
    from ruamel.yaml.comments import CommentedMap
    pl = cfg.get("PHONE_LOCATIONS")
    if not isinstance(pl, dict):
        pl = cfg["PHONE_LOCATIONS"] = CommentedMap()
    prov = pl.get("PROVIDERS")
    if not isinstance(prov, dict):
        prov = pl["PROVIDERS"] = CommentedMap()
    if "DORYAB" not in prov or not isinstance(prov["DORYAB"], dict):
        prov["DORYAB"] = CommentedMap()
    chosen = prov["DORYAB"].get("SRC_SCRIPT")
    prefer = choose_doryab_script(prefer_main=force_main_preference) or chosen
    if prefer and prefer != chosen:
        prov["DORYAB"]["SRC_SCRIPT"] = prefer
    prov["DORYAB"]["COMPUTE"] = True
    return prov["DORYAB"].get("SRC_SCRIPT")

def patch_doryab_required_defaults(cfg):
    pl = _get_sensor_block(cfg, "PHONE_LOCATIONS")
    if not pl or not isinstance(pl.get("PROVIDERS"), dict):
        return
    dory = pl["PROVIDERS"].get("DORYAB")
    if not isinstance(dory, dict):
        return
    defaults = {
        "THRESHOLD_MAX_SPEED": 50,
        "STOP_SPEED": 0.5,
        "STOP_DURATION_MINUTES": 10,
        "RADIUS_METERS": 100,
        "MIN_SAMPLES": 5,
    }
    for k, v in defaults.items():
        dory.setdefault(k, v)
    if "FEATURES" not in dory or not isinstance(dory["FEATURES"], list) or len(dory["FEATURES"]) == 0:
        dory["FEATURES"] = flow_seq(["daily_RR0SS"])

# ---------------- LOCMAP helpers ----------------
def _circle_polygon(lat, lon, radius_m, n=64):
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
    pts = []
    for k in range(n):
        ang = 2*math.pi*k/n
        pts.append([lon + dlon*math.cos(ang), lat + dlat*math.sin(ang)])
    pts.append(pts[0])
    return {"type":"Polygon","coordinates":[pts]}

def _canon_label(s):
    s = (s or "").strip().lower()
    alias = {
        "work":"study", "office":"study", "uni":"study", "university":"study", "school":"study",
        "gym":"exercise",
        "park":"greens", "green":"greens", "greenspace":"greens",
        "house":"home", "apartment":"home", "flat":"home", "residence":"home",
        "living":"living",
    }
    return alias.get(s, s)

# ---- LOCMAP from locations.csv (no sklearn; grid-based clustering) ----
def _grid_cluster(lat_np, lon_np, eps_m):
    """Return integer cell per point using a lat/lon grid sized by eps_m at median latitude."""
    import numpy as np, math
    EARTH_M_PER_DEG = 111320.0
    lat0 = float(np.median(lat_np)) if len(lat_np) else 0.0
    dlat = eps_m / EARTH_M_PER_DEG
    dlon = eps_m / (EARTH_M_PER_DEG * max(1e-6, math.cos(math.radians(lat0))))
    ilat = np.floor((lat_np - lat_np.min())/dlat).astype(np.int64)
    ilon = np.floor((lon_np - lon_np.min())/dlon).astype(np.int64)
    return (ilat << 32) + (ilon & 0xffffffff)

def _assign_loc_labels(df, clusters, tzname):
    """Heuristically assign labels to cluster ids based on time-of-day and counts."""
    import numpy as np, pandas as pd, math
    df = df.copy()
    df["cluster"] = clusters
    df = df[df["cluster"] >= 0]
    if df.empty:
        return {}, {}

    ts_local = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(tzname)
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.dayofweek  # Mon=0
    df["is_wkdy"] = df["dow"] < 5

    grp = df.groupby("cluster")
    stats = grp.agg(
        n=("cluster","size"),
        lat=("double_latitude","median"),
        lon=("double_longitude","median"),
        night=("hour",    lambda s: ((s >= 0)  & (s < 6)).sum()),
        worktime=("hour", lambda s: ((s >= 9)  & (s < 17)).sum()),
        evening=("hour",  lambda s: ((s >= 18) & (s < 24)).sum()),
        morning=("hour",  lambda s: ((s >= 5)  & (s < 9)).sum()),
        wkdy=("is_wkdy",  lambda s: int(s.sum())),
        wkend=("is_wkdy", lambda s: int((~s).sum())),
    ).reset_index()

    # HOME: max night (fallback max n)
    if (stats["night"] > 0).any():
        home_row = stats.sort_values(["night","n"], ascending=False).iloc[0]
    else:
        home_row = stats.sort_values("n", ascending=False).iloc[0]
    home_id = int(home_row["cluster"])

    # distances to home (tie-break)
    def _haversine_m(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        return 2.0 * 6371000.0 * np.arcsin(np.sqrt(a))

    stats["dist_home_m"] = _haversine_m(stats["lat"], stats["lon"], home_row["lat"], home_row["lon"])

    assigned, used = {"home": home_id}, {home_id}

    # STUDY: weekday 9–17 + distance
    cand = stats[~stats["cluster"].isin(used)].copy()
    if not cand.empty:
        cand["score"] = cand["worktime"] + 0.1*cand["wkdy"] + 0.001*cand["n"] + 0.0005*cand["dist_home_m"]
        sr = cand.sort_values("score", ascending=False).iloc[0]
        if sr["worktime"] > 0:
            assigned["study"] = int(sr["cluster"]); used.add(int(sr["cluster"]))

    # LIVING: evenings / second night
    cand = stats[~stats["cluster"].isin(used)].copy()
    if not cand.empty:
        cand["score"] = cand["evening"] + 0.1*cand["night"] + 0.01*cand["n"]
        lr = cand.sort_values("score", ascending=False).iloc[0]
        if lr["evening"] > 0 or lr["night"] > 0:
            assigned["living"] = int(lr["cluster"]); used.add(int(lr["cluster"]))

    # EXERCISE: morning+evening spikes
    cand = stats[~stats["cluster"].isin(used)].copy()
    if not cand.empty:
        cand["score"] = (cand["morning"] + cand["evening"]) - 0.001*cand["n"]
        er = cand.sort_values("score", ascending=False).iloc[0]
        if (er["morning"] + er["evening"]) > 0:
            assigned["exercise"] = int(er["cluster"]); used.add(int(er["cluster"]))

    # GREENS: far from home + weekend
    cand = stats[~stats["cluster"].isin(used)].copy()
    if not cand.empty:
        cand["score"] = 0.001*cand["dist_home_m"] + 0.01*cand["wkend"]
        gr = cand.sort_values("score", ascending=False).iloc[0]
        assigned["greens"] = int(gr["cluster"])

    centroids = { int(r["cluster"]): (float(r["lat"]), float(r["lon"])) for _, r in stats.iterrows() }
    return assigned, centroids

# ---------------- main ----------------
def main():
    yaml = get_yaml()
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["participants","features"], required=True)
    ap.add_argument(
        "--pids",
        default=None,
        help="Comma-separated list of PIDs. If omitted, all PIDs are read from participant_files.csv."
    )
    ap.add_argument("--aware-folder", default="data/external/aware_csv")
    ap.add_argument("--participants-dir", default="data/external/participant_files")
    ap.add_argument("--tz", default=None)
    ap.add_argument("--locations-provider", choices=["barnett","doryab","both"], default="both")
    ap.add_argument("--locations-to-use", dest="locations_to_use", default="ALL",
                        help="Set PHONE_LOCATIONS.LOCATIONS_TO_USE (e.g., ALL, FUSED, FUSED_RESAMPLED, RAW). Case-insensitive.")
    args = ap.parse_args()

    # PIDs
    if args.pids:
        pids = [p.strip() for p in args.pids.split(",") if p.strip()]
        print(f"[info] Using PIDs from CLI: {pids}")
    else:
        pids = load_pids_from_participant_csv(PARTICIPANT_CSV_DEFAULT)

    if pids:
        set_inline_list(cfg, "PIDS", pids)

    # TIMEZONE
    if args.tz:
        tz = ensure_map(cfg, "TIMEZONE")
        tz["TYPE"] = "SINGLE"
        single = ensure_map(tz, "SINGLE")
        single["TZCODE"] = args.tz
        single["CODE"] = args.tz

    # PHONE_DATA_STREAMS -> aware_csv
    pds = ensure_map(cfg, "PHONE_DATA_STREAMS")
    pds["USE"] = "aware_csv"
    aware_csv = ensure_map(pds, "aware_csv")
    aware_csv["FOLDER"] = args.aware_folder

    # Participant files
    paths = ensure_map(cfg, "PATHS")
    paths["PARTICIPANT_FILES"] = args.participants_dir
    cpf = ensure_map(cfg, "CREATE_PARTICIPANT_FILES")
    cpf["CSV_FILE_PATH"] = f"{args.participants_dir}/participant_files.csv"
    cpf["OUTPUT_FOLDER"] = args.participants_dir
    for sec, add in (("PHONE_SECTION", True), ("FITBIT_SECTION", False), ("EMPATICA_SECTION", False)):
        s = ensure_map(cpf, sec)
        s["ADD"] = add
        if "IGNORED_DEVICE_IDS" not in s or not isinstance(s["IGNORED_DEVICE_IDS"], list):
            s["IGNORED_DEVICE_IDS"] = flow_seq([])

    if args.mode == "participants":
        cpf["ENABLED"] = True

    if args.mode == "features":
        cpf["ENABLED"] = False

        # TIME SEGMENTS
        ts = ensure_map(cfg, "TIME_SEGMENTS")
        ts["TYPE"] = "PERIODIC"
        ts["FILE"] = "data/time_segments.csv"
        ts["INCLUDE_PAST_PERIODIC_SEGMENTS"] = False
        print("[info] TIME_SEGMENTS.FILE set to data/time_segments.csv")

        # PHONE_DATA_YIELD
        pdy = cfg.get("PHONE_DATA_YIELD")
        if isinstance(pdy, dict):
            set_inline_list(
                pdy, "SENSORS",
                ["PHONE_SCREEN","PHONE_LOCATIONS","PHONE_BLUETOOTH","PHONE_WIFI_VISIBLE","PHONE_CALLS"]
            )
            provs = pdy.get("PROVIDERS")
            if isinstance(provs, dict):
                from ruamel.yaml.comments import CommentedMap
                if "RAPIDS" not in provs or not isinstance(provs["RAPIDS"], dict):
                    provs["RAPIDS"] = CommentedMap()
                provs["RAPIDS"]["COMPUTE"] = True

        # Disable sensors you don't have
        for k in list(cfg.keys()):
            if isinstance(k, str) and k.startswith("FITBIT_"):
                disable_all(cfg, k)
        for k in ["PHONE_ACTIVITY_RECOGNITION","PHONE_CONVERSATION","PHONE_APPLICATIONS_FOREGROUND",
                  "PHONE_LIGHT","PHONE_MESSAGES","PHONE_WIFI_CONNECTED"]:
            disable_all(cfg, k)

        # Enable core sensors
        set_compute_block(cfg, "PHONE_SCREEN",        prefer="RAPIDS")
        set_compute_block(cfg, "PHONE_CALLS",         prefer="RAPIDS")
        set_compute_block(cfg, "PHONE_BLUETOOTH",     prefer="RAPIDS")
        set_compute_block(cfg, "PHONE_WIFI_VISIBLE",  prefer="RAPIDS")

        # Locations selection
        if args.locations_provider == "barnett":
            set_compute_block(cfg, "PHONE_LOCATIONS", only={"BARNETT"})
        elif args.locations_provider == "doryab":
            dscript = ensure_doryab_provider(cfg, force_main_preference=True)
            set_provider(cfg, "PHONE_LOCATIONS", "BARNETT", COMPUTE=False)
            set_provider(cfg, "PHONE_LOCATIONS", "DORYAB", COMPUTE=True)
            print(f"[info] Using DORYAB for PHONE_LOCATIONS with SRC_SCRIPT={dscript}")
        else:
            set_provider(cfg, "PHONE_LOCATIONS", "BARNETT", COMPUTE=True)
            dscript = ensure_doryab_provider(cfg, force_main_preference=True)
            print(f"[info] Using BOTH providers for PHONE_LOCATIONS (DORYAB script={dscript})")

        # Containers
        set_container(cfg, "PHONE_SCREEN",       "screen.csv")
        set_container(cfg, "PHONE_CALLS",        "calls.csv")
        set_container(cfg, "PHONE_BLUETOOTH",    "bluetooth.csv")
        set_container(cfg, "PHONE_WIFI_VISIBLE", "wifi_visible.csv")
        set_container(cfg, "PHONE_LOCATIONS",    "locations.csv")

        # Minimal provider keys
        set_provider(cfg, "PHONE_CALLS", "RAPIDS", FEATURES_TYPE="EPISODES")

        # LOCATIONS_TO_USE override (e.g., ALL / FUSED / FUSED_RESAMPLED / RAW)
        pl = ensure_map(cfg, "PHONE_LOCATIONS")
        pl["LOCATIONS_TO_USE"] = "ALL"
        pl["ACCURACY_LIMIT"] = 0
        print(f"[info] PHONE_LOCATIONS.LOCATIONS_TO_USE -> {pl}")

        # Generic SRC_SCRIPTs
        set_src(cfg, "PHONE_SCREEN",       "RAPIDS",  "/rapids/src/features/entry.py")
        set_src(cfg, "PHONE_CALLS",        "RAPIDS",  "/rapids/src/features/entry.R")
        set_src(cfg, "PHONE_BLUETOOTH",    "RAPIDS",  "/rapids/src/features/entry.R")
        set_src(cfg, "PHONE_WIFI_VISIBLE", "RAPIDS",  "/rapids/src/features/entry.R")
        _warn_if_missing(["/rapids/src/features/entry.R","/rapids/src/features/entry.py"])

        # Refine SRC_SCRIPTs from tree
        changes = refine_src_scripts_from_tree(cfg, collect_changes=True)
        if changes:
            print("[info] Refined SRC_SCRIPT from tree for:")
            for row in changes:
                print("   ", row)

        # Enforce Barnett script and ensure Doryab defaults
        enforce_barnett_script(cfg)
        d_final = ensure_doryab_provider(cfg, force_main_preference=True)
        patch_doryab_required_defaults(cfg)
        if d_final:
            print(f"[info] DORYAB SRC_SCRIPT now -> {d_final}")

        # --- Bluetooth: ensure DORYAB is enabled in addition to RAPIDS ---
        set_provider(cfg, "PHONE_BLUETOOTH", "DORYAB", COMPUTE=True)
        bt_d = find_provider_script("PHONE_BLUETOOTH", "DORYAB")
        if bt_d:
            set_src(cfg, "PHONE_BLUETOOTH", "DORYAB", bt_d)
            print(f"[info] PHONE_BLUETOOTH.DORYAB SRC_SCRIPT -> {bt_d}")

        # Whitelist (final authority)
        allowed_locations = (
            {"BARNETT"} if args.locations_provider == "barnett"
            else {"DORYAB"} if args.locations_provider == "doryab"
            else {"BARNETT", "DORYAB"}
        )
        apply_whitelist(cfg, allowed={
            "PHONE_SCREEN":       {"RAPIDS"},
            "PHONE_CALLS":        {"RAPIDS"},
            "PHONE_BLUETOOTH":    {"RAPIDS", "DORYAB"},   # <- include DORYAB
            "PHONE_WIFI_VISIBLE": {"RAPIDS"},
            "PHONE_LOCATIONS":    allowed_locations,
        })

        # Locations providers: Barnett + Doryab
        pl   = ensure_map(cfg, "PHONE_LOCATIONS")
        lpro = ensure_map(pl,  "PROVIDERS")
        for prov_name in ("BARNETT","DORYAB"):
            p = ensure_map(lpro, prov_name)
            p["COMPUTE"] = True
            p["USE_LOCATION_MAP"] = False

        # Screen provider: RAPIDS
        ps   = ensure_map(cfg, "PHONE_SCREEN")
        spro = ensure_map(ps,  "PROVIDERS")
        scr  = ensure_map(spro, "RAPIDS")
        scr["COMPUTE"] = True
        scr["USE_LOCATION_MAP"] = False

        print("[info] Enabled USE_LOCATION_MAP for Locations (Barnett/Doryab) + Screen/RAPIDS.")

        # Optional: keep MODEL_NAMES empty if present
        if "MODEL_NAMES" in cfg:
            set_inline_list(cfg, "MODEL_NAMES", [])

    # ---- write back config ----
    with open(CFG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

if __name__ == "__main__":
    main()
