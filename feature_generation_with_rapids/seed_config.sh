#!/usr/bin/env bash
set -euo pipefail

# --------- EDIT THESE IF NEEDED ----------
export BASE='/c/Users/Nirav Bavadiya/Documents/study/Masters/Thesis/feature_generation_thesis'
export RIMG='moshiresearch/rapids:latest'
export TZSTR='Europe/Berlin'
export CNAME='aware_container'
# -----------------------------------------

# If not set by caller, fall back to a default folder (useful for manual testing)
if [ -z "${AWARE_CHUNK_DIR:-}" ]; then
  AWARE_CHUNK_DIR="$BASE/raw_data/2025-10-13_2025-10-19"
fi

echo "[info] Raw AWARE folder on host: $AWARE_CHUNK_DIR"


# Prevent Git Bash path mangling issues with colon/comma args (safe to set)
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

PARTS_CSV="$BASE/rapids_out/external/participant_files/participant_files.csv"
TS_CSV="$BASE/rapids_out/time_segments.csv"
PATCH_PY="$BASE/1_patch_config.py"
EXPORT_OUT_PATH="$BASE/raw_data"

# Helper: interactive exec (Git Bash → winpty)
EXEC="winpty docker exec -it $CNAME bash -lc"


# Sanity checks (we do NOT create/copy these)
[ -f "$PARTS_CSV" ] || { echo "[error] Missing $PARTS_CSV"; exit 1; }
[ -f "$TS_CSV" ]   || { echo "[error] Missing $TS_CSV"; exit 1; }
[ -f "$PATCH_PY" ] || { echo "[error] Missing $PATCH_PY"; exit 1; }

echo "[ok] Found participant CSV: $PARTS_CSV"
echo "[ok] Found time segments   : $TS_CSV"
echo "[ok] Found patch_config.py : $PATCH_PY"

# Recreate container each run so mounts reflect current BASE contents
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "[info] Creating container $CNAME from $RIMG"

# Use --mount (plays nicer with spaces) for all three binds
docker run -d --name "$CNAME" \
  -e TZ="$TZSTR" \
  -m 48g --memory-swap 72g \
  --cpus 8 \
  --shm-size 16g \
  --mount "type=bind,source=$BASE,target=/workspace_feature_generation_thesis" \
  --mount "type=bind,source=$BASE/rapids_out,target=/rapids/data" \
  --mount "type=bind,source=$AWARE_CHUNK_DIR,target=/rapids/data/external/aware_csv,readonly" \
  "$RIMG" bash -lc "sleep infinity" >/dev/null

# Verify /workspace is mounted; if not, fall back to docker cp
if $EXEC 'test -d /workspace_feature_generation_thesis'; then
  $EXEC 'echo "[info] /workspace_feature_generation_thesis is mounted:"; ls -la /workspace_feature_generation_thesis | sed -n "1,120p"'
else
  echo "[warn] /workspace_feature_generation_thesis bind not visible in container — falling back to copy"
  $EXEC 'mkdir -p /workspace_feature_generation_thesis'
  # Copy EVERYTHING currently in BASE into /workspace (one-time snapshot per run)
  docker cp "$BASE/." "$CNAME:/workspace_feature_generation_thesis/"
  $EXEC 'echo "[info] /workspace_feature_generation_thesis contents after copy:"; ls -la /workspace_feature_generation_thesis | sed -n "1,120p"'
fi

# Seed config.yaml once if missing (inside container; lives in /rapids/data)
$EXEC 'test -f /rapids/data/config.yaml || cp /rapids/example_profile/example_config.yaml /rapids/data/config.yaml; echo "config.yaml ready"'

# Ensure ruamel.yaml is installed in THIS container
$EXEC '
python - <<PY
import sys, subprocess
try:
    import ruamel.yaml
    print("ruamel.yaml OK")
except Exception:
    print("Installing ruamel.yaml ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "ruamel.yaml"])
    import ruamel.yaml
    print("ruamel.yaml installed")
PY
'

# Derive comma-separated PIDs from your participants CSV (assumes 4th column is 'pid')
# but allow caller to override via env PIDS (e.g. PIDS="P001")
if [ -z "${PIDS:-}" ]; then
  PIDS="$(awk -F, 'NR>1 && $4!="" {print $4}' "$PARTS_CSV" | sort -u | paste -sd, -)"
  [ -z "$PIDS" ] && PIDS="P001"
fi
echo "[info] Using PIDS: $PIDS"


# Snakemake file to build participant YAMLs
$EXEC 'cat > /rapids/data/participants_only.smk << "SMK"
configfile: "/rapids/data/config.yaml"
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"
rule all:
    input:
        expand("data/external/participant_files/{pid}.yaml", pid=config.get("PIDS", []))
SMK
ls -l /rapids/data/participants_only.smk'

# Use your patch_config.py from /workspace (mounted or copied)
PATCH_PATH="/workspace_feature_generation_thesis/1_patch_config.py"

# Patch config for PARTICIPANTS
$EXEC "python $PATCH_PATH \
  --mode participants \
  --pids '$PIDS' \
  --aware-folder data/external/aware_csv \
  --participants-dir data/external/participant_files \
  --tz $TZSTR && \
  echo 'Config updated for participant generation.' && \
  sed -n '1,150p' /rapids/data/config.yaml"

# Build participant YAMLs using your existing participant_files.csv
$EXEC 'snakemake -s /rapids/data/participants_only.smk -j1 create_participants_files'
$EXEC 'snakemake -s /rapids/data/participants_only.smk -j1'
$EXEC 'echo && echo "--- Generated participant YAMLs ---" && ls -l /rapids/data/external/participant_files/*.yaml'

# Snakemake for features
$EXEC 'cat > /rapids/data/features_only.smk << "SMK"
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

# Patch config for FEATURES (points to your existing data/time_segments.csv)
$EXEC "python $PATCH_PATH \
  --mode features \
  --pids '$PIDS' \
  --aware-folder data/external/aware_csv \
  --participants-dir data/external/participant_files \
  --tz $TZSTR && \
  echo 'Config updated for feature generation.' && \
  awk '/^TIME_SEGMENTS:/,/^[A-Z_]+:/' /rapids/data/config.yaml || true"

# Dry run and real run (first PID)
first_pid="$(echo "$PIDS" | cut -d, -f1)"

echo "[info] Snakemake dry run for PIDS=$PIDS"
$EXEC "cd /rapids && snakemake -s /rapids/data/features_only.smk -n"

echo "[info] Running feature generation for PIDS=$PIDS"
$EXEC "cd /rapids && snakemake -s /rapids/data/features_only.smk --cores 4 --printshellcmds"

echo
echo "[done] Features generated for PIDS=$PIDS (example: $BASE/rapids_out/processed/features/${first_pid}/all_sensor_features.csv)"

LOC_PATH="$AWARE_CHUNK_DIR/locations.csv"
POLY_CSV="$BASE/aware_raw_data/$PIDS/locmap/locmap.csv"
WRITE_CSV="$BASE/rapids_out/processed/features/$PIDS/all_sensor_features.csv"

python 2_locmap_from_polygons.py \
  --pid "$PIDS" \
  --locations "$LOC_PATH" \
  --polygons  "$POLY_CSV" \
  --write_csv    "$WRITE_CSV" \
  --tz "Europe/Berlin"\
  --debug 1


python 3_add_all_epochs_batch_mod.py
python 4_normalize_globem_epochs.py --base "$BASE"
