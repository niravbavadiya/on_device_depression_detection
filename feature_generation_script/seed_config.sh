#!/usr/bin/env bash

# set once per shell
export BASE='/c/Users/Nirav Bavadiya/Documents/study/Masters/Thesis/feature_generation_thesis'
export RIMG='moshiresearch/rapids:latest'

# make working folder (if not already)
mkdir -p "$BASE/rapids_out"
mkdir -p "$BASE/rapids_out/external"
mkdir -p "$BASE/rapids_out/external/participant_files"


# if you don’t yet have a config.yaml, seed it from the image
winpty docker run --rm -it \
  -v "$BASE/rapids_out":/host_out \
  "$RIMG" \
  bash -lc 'test -f /host_out/config.yaml || cp /rapids/example_profile/example_config.yaml /host_out/config.yaml; echo "config.yaml ready"'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" \
  bash -lc '
set -euo pipefail

# 0) Ensure vendor directory exists (persists on host)
mkdir -p /rapids/data/vendor/python

# 1) Write patch_config.py (does NOT import ruamel before setting sys.path)
cat > /rapids/data/patch_config.py << "PY"
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
PY

echo "patch_config.py ready (restart-proof ruamel vendoring)"
'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" \
  bash -lc 'python /rapids/data/patch_config.py \
              --mode participants \
              --pids P001 \
              --aware-folder data/external/aware_csv \
              --participants-dir data/external/participant_files \
              --tz Europe/Berlin && \
            echo "Config updated for participant generation." && \
            sed -n "1,150p" /rapids/data/config.yaml'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" \
  bash -lc 'cat > /rapids/data/participants_only.smk << "SMK"
# Only what we need to create participant YAMLs
configfile: "/rapids/data/config.yaml"
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"

# Build a YAML per PID when you run 'snakemake -s participants_only.smk -j1'
rule all:
    input:
        expand("data/external/participant_files/{pid}.yaml", pid=config.get("PIDS", []))
SMK
ls -l /rapids/data/participants_only.smk'

winpty docker run --rm -it -v "$BASE/rapids_out":/rapids/data "$RIMG" bash -lc '
cat > /rapids/data/external/participant_files/participant_files.csv << "CSV"
device_id,fitbit_id,empatica_id,pid,label,platform,start_date,end_date
0adab909-eb8f-4baf-9eed-7a4cc889cb7c,,,P001,P001,android,,
CSV

nl -ba /rapids/data/external/participant_files/participant_files.csv
'

winpty docker run --rm -it \
  -v "$BASE/raw_data":/rapids/data/external/aware_csv:ro \
  -v "$BASE/rapids_out":/rapids/data \
  -e TZ=Europe/Berlin \
  "$RIMG" \
  bash -lc 'snakemake -s /rapids/data/participants_only.smk -j1 create_participants_files && \
            snakemake -s /rapids/data/participants_only.smk -j1 && \
            echo && echo "--- Generated participant YAMLs ---" && ls -l /rapids/data/external/participant_files/*.yaml'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" bash -lc '
cat > /rapids/data/features_only.smk << "SMK"
configfile: "/rapids/data/config.yaml"
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"
include: "/rapids/rules/features.smk"

_pids = config.get("PIDS") or []

rule all:
    input:
        expand("data/processed/features/{pid}/all_sensor_features.csv", pid=_pids)
SMK
ls -l /rapids/data/features_only.smk'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" \
  bash -lc 'python /rapids/data/patch_config.py \
              --mode features \
              --pids P001 \
              --aware-folder data/external/aware_csv \
              --participants-dir data/external/participant_files \
              --tz Europe/Berlin && \
            echo "Config updated for feature generation." && \
            sed -n "1,200p" /rapids/data/config.yaml'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" \
  bash -lc 'cat > /rapids/data/features_only.smk << "SMK"
configfile: "/rapids/data/config.yaml"

# absolute include paths avoid CWD issues
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"
include: "/rapids/rules/features.smk"

_pids = config.get("PIDS") or []

rule all:
    input:
        expand("data/processed/features/{pid}/all_sensor_features.csv", pid=_pids)
SMK
ls -l /rapids/data/features_only.smk'


winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" bash -lc '
set -euo pipefail

# -- Ensure ruamel.yaml is available in THIS container --
python - <<PY
import sys, subprocess
try:
    import ruamel.yaml
    print("ruamel.yaml OK")
except Exception:
    print("Installing ruamel.yaml in container...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ruamel.yaml"])
    import ruamel.yaml
    print("ruamel.yaml installed")
PY

# 1) Find the R file that defines barnett_features
BARNETT_DIR="/rapids/src/features/phone_locations/barnett"
BARNETT_SCRIPT="$(grep -Rl --include="*.R" "barnett_features[[:space:]]*<-[[:space:]]*function" "$BARNETT_DIR" | head -n1 || true)"
if [ -z "$BARNETT_SCRIPT" ]; then
  echo "[error] Could not find an R file that defines barnett_features under $BARNETT_DIR"
  ls -1 "$BARNETT_DIR"/*.R || true
  exit 1
fi
echo "[info] Using BARNETT SRC_SCRIPT: $BARNETT_SCRIPT"
export BARNETT_SCRIPT

# 2) Patch config.yaml (preserve formatting)
python - <<'"PY"'
import os, io
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

p = "/rapids/data/config.yaml"
yaml = YAML(); yaml.preserve_quotes = True
cfg = yaml.load(io.open(p, "r", encoding="utf-8"))

def ensure_map(m, k):
    if k not in m or not isinstance(m[k], dict):
        m[k] = CommentedMap()
    return m[k]

barnett_path = os.environ["BARNETT_SCRIPT"]
pl   = ensure_map(cfg, "PHONE_LOCATIONS")
prov = ensure_map(pl,  "PROVIDERS")
barn = ensure_map(prov,"BARNETT")
barn["SRC_SCRIPT"] = barnett_path
barn["COMPUTE"]    = True

yaml.dump(cfg, io.open(p, "w", encoding="utf-8"))
print("[ok] config.yaml updated with BARNETT SRC_SCRIPT ->", barnett_path)
PY

# 3) Show the relevant block
awk "/^PHONE_LOCATIONS:/,/^[A-Z_]+:/" /rapids/data/config.yaml
'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" bash -lc '
set -euo pipefail

# 0) Make sure ruamel.yaml is available in the container
python - <<PY || python -m pip install --no-cache-dir ruamel.yaml >/dev/null
import ruamel.yaml; print("ruamel.yaml OK")
PY

# 1) Patch config.yaml: set SRC_SCRIPT for DORYAB and keep COMPUTE true for both
python - <<PY
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import io

p="/rapids/data/config.yaml"
yaml=YAML(); yaml.preserve_quotes=True
cfg=yaml.load(io.open(p,"r",encoding="utf-8"))

def ensure(m,k):
    if k not in m or not isinstance(m[k], dict):
        m[k]=CommentedMap()
    return m[k]

pl   = ensure(cfg, "PHONE_LOCATIONS")
prov = ensure(pl,  "PROVIDERS")
barn = ensure(prov,"BARNETT")
dory = ensure(prov,"DORYAB")

# Barnett should point to an R file that defines barnett_features (you already set main.R earlier)
# Doryab must point to the Python feature entry
dory["SRC_SCRIPT"] = "/rapids/src/features/phone_locations/doryab/main.py"
dory["COMPUTE"]    = True
barn["COMPUTE"]    = True

yaml.dump(cfg, io.open(p,"w",encoding="utf-8"))
print("[ok] Set DORYAB SRC_SCRIPT ->", dory["SRC_SCRIPT"])
PY

# 2) Show the PHONE_LOCATIONS block for sanity
awk "/^PHONE_LOCATIONS:/,/^[A-Z_]+:/" /rapids/data/config.yaml
'

winpty docker run --rm -it \
  -v "$BASE/rapids_out":/rapids/data \
  "$RIMG" bash -lc '
set -euo pipefail

# Ensure ruamel.yaml is available in THIS container
python - <<PY || python -m pip install --no-cache-dir ruamel.yaml >/dev/null
import ruamel.yaml; print("ruamel.yaml OK")
PY

# Patch the DORYAB provider with required params and keep formatting
python - <<PY
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import io

p = "/rapids/data/config.yaml"
yaml = YAML(); yaml.preserve_quotes = True
cfg = yaml.load(io.open(p, "r", encoding="utf-8"))

def ensure_map(m, k):
    if k not in m or not isinstance(m[k], dict):
        m[k] = CommentedMap()
    return m[k]

def flow_list(items):
    s = CommentedSeq(items)
    try:
        s.yaml_set_flow_style()
    except Exception:
        pass
    return s

pl   = ensure_map(cfg, "PHONE_LOCATIONS")
prov = ensure_map(pl,  "PROVIDERS")
dory = ensure_map(prov, "DORYAB")

# Keep your SRC_SCRIPT if already set; otherwise set it.
dory.setdefault("SRC_SCRIPT", "/rapids/src/features/phone_locations/doryab/main.py")
dory["COMPUTE"] = True

# Provide minimal parameters expected by the Doryab code
# (tuned to reasonable defaults; units are meters/second & minutes & meters)
defaults = {
    "THRESHOLD_MAX_SPEED": 50,           # 50 m/s (~180 km/h), filter insane jumps
    "STOP_SPEED": 0.5,                   # m/s threshold for stop detection
    "STOP_DURATION_MINUTES": 10,         # min duration to consider a stop
    "RADIUS_METERS": 100,                # DBSCAN eps / cluster radius
    "MIN_SAMPLES": 5,                    # DBSCAN min samples
}

for k, v in defaults.items():
    dory.setdefault(k, v)

# Ensure FEATURES is present and inline (flow) style
if "FEATURES" not in dory or not isinstance(dory["FEATURES"], list) or len(dory["FEATURES"]) == 0:
    dory["FEATURES"] = flow_list(["daily_RR0SS"])

yaml.dump(cfg, io.open(p, "w", encoding="utf-8"))
print("[ok] DORYAB provider patched with required parameters in", p)
PY

# Show the PHONE_LOCATIONS block (for sanity)
awk "/^PHONE_LOCATIONS:/,/^[A-Z_]+:/" /rapids/data/config.yaml || true
'
winpty docker run --rm -it \
  -v "$BASE/raw_data":/rapids/data/external/aware_csv:ro \
  -v "$BASE/rapids_out":/rapids/data \
  -e TZ=Europe/Berlin \
  "$RIMG" \
  bash -lc 'snakemake -s /rapids/data/features_only.smk -n'

winpty docker run --rm -it \
  -v "$BASE/raw_data":/rapids/data/external/aware_csv:ro \
  -v "$BASE/rapids_out":/rapids/data \
  -e TZ=Europe/Berlin \
  -e PYTHONPATH=/rapids/src:$PYTHONPATH \
  "$RIMG" \
  bash -lc 'cd /rapids && \
    snakemake -s /rapids/data/features_only.smk --cores 4 --printshellcmds -- \
      data/processed/features/P001/all_sensor_features.csv'