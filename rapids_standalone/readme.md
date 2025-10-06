# GLOBEM-style RAPIDS Feature Generation (Standalone)

## Quick start
1) Install Docker & Docker Compose.
2) Edit `profiles/globem/config.yaml`:
   - For MySQL: set HOST/USER/PASSWORD/DB_NAME under `aware_mysql`
   - Or switch to `aware_csv` and drop AWARE CSVs in `data/raw/`
3) Fill `data/external/participants.csv` with your pid, device_id, timezone.
4) Run:
   ```bash
   ./run_rapids.sh
