#!/usr/bin/env python
"""
Labeling Script with Windowing

This script reads sensor data from a sample’s FeatureData folder and survey files
from its SurveyData folder. Using a fixed window (e.g., 14 days), it assigns survey
responses (labels) to sensor records that fall within the window defined around each
survey date. Multiple survey files (dep_weekly, dep_endterm, ema) are handled using
windowing. Pre‐ and post‐survey files (pre, post) are merged based on subject id.

Output:
  A labeled CSV file is written to labeled_data/labeled_sensor.csv.
  
Adjust WINDOW_SIZE as needed.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# CONFIGURATION
SAMPLE_FOLDER = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_pruned\INS-W-sample_1")  # change this to your sample folder path; e.g., INS-W-sample_001
FEATURE_FILE = SAMPLE_FOLDER / "FeatureData" / "norm" / "location.csv"
SURVEY_DIR   = SAMPLE_FOLDER / "SurveyData"
OUTPUT_DIR   = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\4_data_labeled")
OUTPUT_FILE  = OUTPUT_DIR / "labeled_location.csv"

# Define window size in days for time-varying surveys
WINDOW_SIZE = 14  
HALF_WINDOW = WINDOW_SIZE // 2

# List of survey files that are time-dependent (they have a date for each survey)
time_varying_surveys = {
    "dep_weekly": {"file": "dep_weekly.csv", "label_columns": ["dep"]},
    "dep_endterm": {"file": "dep_endterm.csv", "label_columns": ["dep", "BDI2"]},
    "ema": {"file": "ema.csv", "label_columns": ["negative_affect_EMA"]}
}
# Static surveys (pre and post) will be merged on pid.
static_surveys = {
    "pre": {"file": "pre.csv"},
    "post": {"file": "post.csv"}
}

# ----------------- Helper Function: assign window labels -----------------
def assign_window_labels(sensor_df, survey_df, label_cols):
    """
    Given sensor data with a 'date' column and a survey_df with a 'date' column,
    for each survey row define a window:
       window_start = survey_date - HALF_WINDOW days
       window_end   = survey_date + HALF_WINDOW days.
    Then, for each sensor record, if its date falls into a window, assign the survey's
    label(s). If multiple windows overlap, you may choose to use the most recent survey.
    Here, we will simply backfill by merging asof.
    """
    # Sort dataframes by date.
    sensor_df = sensor_df.sort_values("date").copy()
    survey_df = survey_df.sort_values("date").copy()
    
    # For each label col in label_cols, perform a merge_asof.
    # We use 'backward' so that sensor rows get the most recent survey response.
    for lab in label_cols:
        survey_tmp = survey_df[["pid", "date", lab]].copy()
        # Create a window column: only assign if sensor.date is within half-window after survey date.
        # Here we simulate by merging_asof then filtering out those where the gap is too large.
        sensor_df = pd.merge_asof(sensor_df, survey_tmp,
                                  on="date",
                                  by="pid",
                                  direction="backward",
                                  tolerance=pd.Timedelta(days=HALF_WINDOW))
        # The merged column will have the same name as lab.
    return sensor_df

# ----------------- Main Labeling Pipeline -----------------
def main():
    # 1. Load sensor data.
    sensor_df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    
    # 2. Process each time-varying survey.
    for key, info in time_varying_surveys.items():
        survey_path = SURVEY_DIR / info["file"]
        if not survey_path.exists():
            print(f"Warning: {survey_path} not found. Skipping {key}.")
            continue
        survey_df = pd.read_csv(survey_path, parse_dates=["date"])
        # Merge sensor data with survey data using windowing.
        sensor_df = assign_window_labels(sensor_df, survey_df, info["label_columns"])
    
    # 3. Process static surveys (pre and post); merge on pid.
    for key, info in static_surveys.items():
        survey_path = SURVEY_DIR / info["file"]
        if not survey_path.exists():
            print(f"Warning: {survey_path} not found. Skipping {key}.")
            continue
        survey_df = pd.read_csv(survey_path, parse_dates=["date"])
        # Drop the date column since static surveys occur once.
        survey_df = survey_df.drop(columns=["date"], errors="ignore")
        # Merge sensor data on pid.
        sensor_df = pd.merge(sensor_df, survey_df, on="pid", how="left", suffixes=("", f"_{key}"))
    
    # 4. Optionally, you can fill remaining NaN labels or keep them to later mask in training.
    # For now, we simply write out the labeled sensor data.
    OUTPUT_DIR.mkdir(exist_ok=True)
    sensor_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Labeled sensor data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
