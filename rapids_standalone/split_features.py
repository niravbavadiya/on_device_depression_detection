import pandas as pd
import os
import re

FEATURE_DIR = "output/FeatureData"
RAPIDS_FILE = os.path.join(FEATURE_DIR, "rapids.csv")

# Mapping of RAPIDS column prefixes → modality filenames
PREFIX_MAP = {
    "phone_locations__": "location.csv",
    "phone_screen__": "screen.csv",
    "fitbit_steps__": "steps.csv",
    "fitbit_sleep__": "sleep.csv",
    "phone_bluetooth__": "bluetooth.csv",
    "phone_calls__": "calls.csv",
    "fitbit_hr__": "heartrate.csv"
}

def split_features():
    if not os.path.exists(RAPIDS_FILE):
        print(f"[ERROR] RAPIDS file not found: {RAPIDS_FILE}")
        return

    print(f"[INFO] Loading {RAPIDS_FILE} ...")
    df = pd.read_csv(RAPIDS_FILE)

    # Keep index columns (pid, date, etc.)
    id_cols = [c for c in df.columns if re.match(r"pid|date|participant|user|label", c, re.I)]

    for prefix, filename in PREFIX_MAP.items():
        cols = id_cols + [c for c in df.columns if c.startswith(prefix)]
        if len(cols) <= len(id_cols):
            continue  # no columns found for this modality

        subdf = df[cols].copy()
        out_path = os.path.join(FEATURE_DIR, filename)
        subdf.to_csv(out_path, index=False)
        print(f"[OK] Saved {filename} ({len(subdf.columns) - len(id_cols)} features)")

    print("[DONE] Feature splitting complete.")

if __name__ == "__main__":
    split_features()
