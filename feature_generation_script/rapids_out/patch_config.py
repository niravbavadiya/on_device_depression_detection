from __future__ import annotations
import argparse, os, sys, subprocess, glob

CFG = "/rapids/data/config.yaml"
VENDOR = "/rapids/data/vendor/python"
EXAMPLE_CFG = "/rapids/example_profile/example_config.yaml"

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

def ensure_map(m, k):
    from ruamel.yaml.comments import CommentedMap
    if not isinstance(m, dict) or m is None:
        m = CommentedMap()
    if k not in m or not isinstance(m[k], dict):
        m[k] = CommentedMap()
    return m[k]

def set_inline_list(m, key, items):
    m[key] = flow_seq(items)

def set_compute_block(cfg, sensor_key, prefer: str|None=None, only: set[str]|None=None):
    blk = cfg.get(sensor_key)
    if not isinstance(blk, dict):
        return
    provs = blk.get("PROVIDERS")
    if not isinstance(provs, dict):
        return
    names = list(provs.keys())
    if only and any(n in names for n in only):
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

# ---------- minimal provider/sensor helpers ----------
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

def set_src(cfg, sensor_key, provider, path_abs: str):
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

def set_container(cfg, sensor_key, filename: str):
    blk = _get_sensor_block(cfg, sensor_key)
    if blk is not None:
        blk["CONTAINER"] = filename

def apply_whitelist(cfg, allowed: dict[str,set[str]]):
    """Final pass: set COMPUTE=True only for allowed providers on phone sensors."""
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
# ---------------------------------------------------

def _warn_if_missing(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("[warn] SRC_SCRIPT target(s) not found: " + ", ".join(missing), file=sys.stderr)

# -------- Patch 15 helpers: discover per-provider scripts ----------
def find_provider_script(sensor_key: str, provider_key: str):
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

def refine_src_scripts_from_tree(cfg, collect_changes: bool = False):
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
    # PHONE_CALLS: ensure FEATURES_TYPE default
    if "PHONE_CALLS" in cfg:
        provs = cfg["PHONE_CALLS"].get("PROVIDERS", {})
        rapids = provs.get("RAPIDS")
        if isinstance(rapids, dict) and "FEATURES_TYPE" not in rapids:
            rapids["FEATURES_TYPE"] = "EPISODES"
            if collect_changes:
                changes.append(("PHONE_CALLS","RAPIDS","FEATURES_TYPE=EPISODES"))
    return changes
# ------------------------------------------------------------------

# -------- Doryab helpers (no longer disable Barnett) --------------
def choose_doryab_script():
    candidates = [
        "/rapids/src/features/phone_locations/doryab/features.py",
        "/rapids/src/features/phone_locations/doryab/provider.py",
        "/rapids/src/features/phone_locations/doryab/compute_features.py",
        "/rapids/src/features/phone_locations/doryab/feature_extraction.py",
        "/rapids/src/features/phone_locations/doryab/main.py",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    any_py = [p for p in glob.glob("/rapids/src/features/phone_locations/doryab/*.py")
              if os.path.basename(p) != "add_doryab_extra_columns.py"]
    return any_py[0] if any_py else None

def ensure_doryab_provider(cfg):
    """Ensure PHONE_LOCATIONS.DORYAB exists, has SRC_SCRIPT, and COMPUTE=True."""
    from ruamel.yaml.comments import CommentedMap
    pl = cfg.get("PHONE_LOCATIONS")
    if not isinstance(pl, dict):
        pl = cfg["PHONE_LOCATIONS"] = CommentedMap()
    prov = pl.get("PROVIDERS")
    if not isinstance(prov, dict):
        prov = pl["PROVIDERS"] = CommentedMap()
    if "DORYAB" not in prov or not isinstance(prov["DORYAB"], dict):
        prov["DORYAB"] = CommentedMap()
    chosen = prov["DORYAB"].get("SRC_SCRIPT") or choose_doryab_script() or "/rapids/src/features/entry.py"
    prov["DORYAB"]["SRC_SCRIPT"] = chosen
    prov["DORYAB"]["COMPUTE"] = True
    return chosen
# ------------------------------------------------------------------

def main():
    yaml = get_yaml()
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["participants","features"], required=True)
    ap.add_argument("--pids", default="P001")
    ap.add_argument("--aware-folder", default="data/external/aware_csv")
    ap.add_argument("--participants-dir", default="data/external/participant_files")
    ap.add_argument("--tz", default=None)  # e.g., Europe/Berlin
    ap.add_argument("--locations-provider", choices=["barnett","doryab","both"], default="both",
                    help="Which provider(s) to use for PHONE_LOCATIONS.")
    args = ap.parse_args()

    # PIDS inline list
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    if pids:
        set_inline_list(cfg, "PIDS", pids)

    # TIMEZONE
    if args.tz:
        tz = ensure_map(cfg, "TIMEZONE")
        tz["TYPE"] = "SINGLE"
        single = ensure_map(tz, "SINGLE")
        single["TZCODE"] = args.tz
        single["CODE"] = args.tz

    # PHONE_DATA_STREAMS aware_csv path
    pds = ensure_map(cfg, "PHONE_DATA_STREAMS")
    pds["USE"] = "aware_csv"
    aware_csv = ensure_map(pds, "aware_csv")
    aware_csv["FOLDER"] = args.aware_folder

    # participant files
    paths = ensure_map(cfg, "PATHS")
    paths["PARTICIPANT_FILES"] = args.participants_dir
    cpf = ensure_map(cfg, "CREATE_PARTICIPANT_FILES")
    cpf["CSV_FILE_PATH"] = f"{args.participants_dir}/participant_files.csv"
    cpf["OUTPUT_FOLDER"] = args.participants_dir

    for sec, add in (("PHONE_SECTION", True), ("FITBIT_SECTION", False), ("EMPATICA_SECTION", False)):
        s = ensure_map(cpf, sec)
        s["ADD"] = add
        if not isinstance(s.get("IGNORED_DEVICE_IDS"), list):
            s["IGNORED_DEVICE_IDS"] = flow_seq([])

    if args.mode == "participants":
        cpf["ENABLED"] = True

    if args.mode == "features":
        cpf["ENABLED"] = False

        # 0) PHONE_DATA_YIELD: inline sensors + RAPIDS compute true
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

        # 1) Disable everything you don’t have
        for k in list(cfg.keys()):
            if isinstance(k, str) and k.startswith("FITBIT_"):
                disable_all(cfg, k)
        for k in [
            "PHONE_ACTIVITY_RECOGNITION",
            "PHONE_CONVERSATION",
            "PHONE_APPLICATIONS_FOREGROUND",
            "PHONE_LIGHT",
            "PHONE_MESSAGES",
            "PHONE_WIFI_CONNECTED",
        ]:
            disable_all(cfg, k)

        # 2) Enable the five sensors (locations depends on flag)
        set_compute_block(cfg, "PHONE_SCREEN",        prefer="RAPIDS")
        set_compute_block(cfg, "PHONE_CALLS",         prefer="RAPIDS")
        set_compute_block(cfg, "PHONE_BLUETOOTH",     prefer="RAPIDS")
        set_compute_block(cfg, "PHONE_WIFI_VISIBLE",  prefer="RAPIDS")

        # Locations providers:
        # - barnett: only Barnett true
        # - doryab:  ensure Doryab exists and true
        # - both (default): ensure BOTH true
        if args.locations_provider == "barnett":
            set_compute_block(cfg, "PHONE_LOCATIONS", only={"BARNETT"})
        elif args.locations_provider == "doryab":
            chosen = ensure_doryab_provider(cfg)
            print(f"[info] Using DORYAB for PHONE_LOCATIONS with SRC_SCRIPT={chosen}")
            # set Doryab true; leave Barnett as-is (may be false/true depending on config)
            set_provider(cfg, "PHONE_LOCATIONS", "DORYAB", COMPUTE=True)
        else:  # both
            # ensure both providers exist & are COMPUTE=True
            set_provider(cfg, "PHONE_LOCATIONS", "BARNETT", COMPUTE=True)
            chosen = ensure_doryab_provider(cfg)  # sets DORYAB true + SRC_SCRIPT
            print(f"[info] Using BOTH providers for PHONE_LOCATIONS (DORYAB script={chosen})")
            # ensure Barnett SRC_SCRIPT generic if missing
            set_src(cfg, "PHONE_LOCATIONS", "BARNETT", "/rapids/src/features/entry.R")

        # 3) Container filenames
        set_container(cfg, "PHONE_SCREEN",       "screen.csv")
        set_container(cfg, "PHONE_CALLS",        "calls.csv")
        set_container(cfg, "PHONE_BLUETOOTH",    "bluetooth.csv")
        set_container(cfg, "PHONE_WIFI_VISIBLE", "wifi_visible.csv")
        set_container(cfg, "PHONE_LOCATIONS",    "locations.csv")

        # 4) Minimal provider keys
        set_provider(cfg, "PHONE_CALLS", "RAPIDS", FEATURES_TYPE="EPISODES")

        # 5) Absolute SRC_SCRIPT paths (generic entry points first)
        set_src(cfg, "PHONE_SCREEN",       "RAPIDS",  "/rapids/src/features/entry.py")
        set_src(cfg, "PHONE_CALLS",        "RAPIDS",  "/rapids/src/features/entry.R")
        set_src(cfg, "PHONE_BLUETOOTH",    "RAPIDS",  "/rapids/src/features/entry.R")
        set_src(cfg, "PHONE_WIFI_VISIBLE", "RAPIDS",  "/rapids/src/features/entry.R")
        # Only set Barnett generic if we are not already refining from tree below
        if args.locations_provider in ("barnett", "both"):
            set_src(cfg, "PHONE_LOCATIONS", "BARNETT", "/rapids/src/features/entry.R")
        _warn_if_missing([
            "/rapids/src/features/entry.R",
            "/rapids/src/features/entry.py",
        ])

        # 5b) Patch 15: refine SRC_SCRIPTs from /rapids/src/features/{sensor}/{provider}/
        changes = refine_src_scripts_from_tree(cfg, collect_changes=True)
        if changes:
            print("[info] Refined SRC_SCRIPT from tree for:")
            for row in changes:
                print("   ", row)

        # 6) Provider whitelist (final authority) — includes both when requested
        allowed_locations = (
            {"BARNETT"} if args.locations_provider == "barnett"
            else {"DORYAB"} if args.locations_provider == "doryab"
            else {"BARNETT", "DORYAB"}
        )
        apply_whitelist(cfg, allowed={
            "PHONE_SCREEN":       {"RAPIDS"},
            "PHONE_CALLS":        {"RAPIDS"},
            "PHONE_BLUETOOTH":    {"RAPIDS"},
            "PHONE_WIFI_VISIBLE": {"RAPIDS"},
            "PHONE_LOCATIONS":    allowed_locations,
        })

        # Optional: keep MODEL_NAMES empty if present
        if "MODEL_NAMES" in cfg:
            set_inline_list(cfg, "MODEL_NAMES", [])

    with open(CFG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

if __name__ == "__main__":
    main()
