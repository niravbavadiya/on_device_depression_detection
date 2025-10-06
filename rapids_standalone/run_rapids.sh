#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting RAPIDS feature generation (GLOBEM-style)..."

# 1️⃣ Start Docker container
docker compose up -d rapids

# 2️⃣ Run RAPIDS inside the container
docker exec rapids_globem bash -lc "
  echo '[RAPIDS] Running feature pipeline...'
  rapids -j1 --profile profiles/globem
"

# 3️⃣ Ensure output folder exists inside container
docker exec rapids_globem bash -lc "mkdir -p /rapids/output/FeatureData"

# 4️⃣ Copy RAPIDS result files to FeatureData
docker exec rapids_globem bash -lc "
  cp -r results/* /rapids/output/FeatureData/ || true
"

# 5️⃣ Run Python feature splitter inside container
echo "[INFO] Splitting RAPIDS output into modality CSVs..."
docker exec rapids_globem bash -lc "python split_features.py || true"

# 6️⃣ Show results
echo
echo "✅ Feature generation complete. Files saved to:"
docker exec rapids_globem bash -lc "ls -lh /rapids/output/FeatureData/"
echo
echo "🎯 On your PC: C:\\Users\\Nirav Bavadiya\\Documents\\study\\Masters\\Thesis\\rapids_standalone\\output\\FeatureData"
