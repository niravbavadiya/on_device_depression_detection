#!/usr/bin/env bash
set -euo pipefail

#python 0_export_aware_to_csv.py \
#  --host 91.98.172.113 \
#  --user root \
#  --password rootpassword \
#  --db aware \
#  --port 3306 \

BASE='/c/Users/Nirav Bavadiya/Documents/study/Masters/Thesis/feature_generation_thesis'
AWARE_ROOT="$BASE/aware_raw_data"
RUN_SCRIPT="$BASE/seed_config.sh"   # rename if your main script is different

export BASE  # so the inner script sees the same BASE

for pid_dir in "$AWARE_ROOT"/*/; do
  [[ -d "$pid_dir" ]] || continue
  pid="${pid_dir%/}"; pid="${pid##*/}"   # e.g., P001

  for week_dir in "$pid_dir"*/; do
    [[ -d "$week_dir" ]] || continue
    week="${week_dir%/}"; week="${week##*/}"  # e.g., 2025-10-13_2025-10-19

    # Skip polygon folders or anything not a week chunk
    if [[ "$week" == "locmap" || "$week" == "_locmap" ]]; then
      echo "[skip] $pid/$week (polygon folder)"
      continue
    fi
    if [[ ! "$week" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
      echo "[skip] $pid/$week (not a week chunk)"
      continue
    fi

    echo
    echo "======================================================="
    echo "[info] Running RAPIDS for PID=$pid, WEEK=$week"
    echo "  raw folder: $week_dir"
    echo "======================================================="

    # 1) Tell seed_config.sh which raw-data folder + PID to use
    export AWARE_CHUNK_DIR="$week_dir"
    export PIDS="$pid"

    # 2) Run your existing RAPIDS feature-generation pipeline
    bash "$RUN_SCRIPT"

    # 3) Optional: snapshot weekly features so they don't overwrite each other
    src="$BASE/rapids_out/processed/features/$pid/all_sensor_features.csv"
    if [ -f "$src" ]; then
      dst_dir="$BASE/rapids_out/processed/features_weekly/$pid"
      mkdir -p "$dst_dir"
      dst="$dst_dir/${week}_all_sensor_features.csv"
      cp "$src" "$dst"
      echo "[ok] Saved weekly features to: $dst"
    else
      echo "[warn] Features file not found for PID=$pid WEEK=$week: $src"
    fi
  done
done

echo
echo "[done] RAPIDS run for all pid/week chunks."
