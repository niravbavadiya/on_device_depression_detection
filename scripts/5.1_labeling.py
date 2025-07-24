import pandas as pd
import os
from datetime import timedelta

# --- Step 1: Load Data ---
location_df = pd.read_csv(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_pruned\INS-W-sample_1\FeatureData\norm\location.csv", parse_dates=["date"])
dep_weekly = pd.read_csv(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_pruned\INS-W-sample_1\SurveyData\dep_weekly.csv", parse_dates=["date"])
dep_end = pd.read_csv(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\data_pruned\INS-W-sample_1\SurveyData\dep_endterm.csv", parse_dates=["date"])

# --- Step 2: Normalize Column Names ---
location_df.rename(columns=lambda x: x.strip().lower(), inplace=True)
dep_weekly.rename(columns=lambda x: x.strip().lower(), inplace=True)
dep_end.rename(columns=lambda x: x.strip().lower(), inplace=True)

# --- Step 3: Sort DataFrames by 'pid' and 'date' ---
location_df = location_df.sort_values(["pid", "date"]).reset_index(drop=True)
dep_weekly = dep_weekly.sort_values(["pid", "date"]).reset_index(drop=True)
dep_end = dep_end.sort_values(["pid", "date"]).reset_index(drop=True)

# --- Step 4: Label dep_3day using merge_asof for each 'pid' ---
def label_dep3day(sensor_df, survey_df):
    # Process each group (user) separately to ensure the "date" column is sorted
    def merge_group(group):
        pid_val = group["pid"].iloc[0]
        # Subset survey records for the current pid; ensure they are sorted by date
        survey_group = survey_df[survey_df["pid"] == pid_val].sort_values("date")
        # Merge asof on the group (sorted by date)
        merged_group = pd.merge_asof(
            group.sort_values("date"),
            survey_group[["date", "dep"]],
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=3)
        )
        return merged_group

    labeled_df = sensor_df.groupby("pid", group_keys=False).apply(merge_group)
    labeled_df.rename(columns={"dep": "dep_3day"}, inplace=True)
    return labeled_df

labeled = label_dep3day(location_df, dep_weekly)

# --- Step 5: Label dep_end using the last 3 days preceding each user's end survey ---
def label_dep_end(df, end_df):
    df = df.copy()
    df["dep_end"] = None

    for pid, group in df.groupby("pid"):
        if pid in end_df["pid"].values:
            # Convert end_date to pd.Timestamp for proper datetime arithmetic
            end_date = pd.Timestamp(end_df.loc[end_df["pid"] == pid, "date"].values[0])
            dep_value = end_df.loc[end_df["pid"] == pid, "dep"].values[0]
            # Mark the days within the window [end_date - 2 days, end_date] for the current pid
            mask = (df["pid"] == pid) & (df["date"] >= end_date - pd.Timedelta(days=2)) & (df["date"] <= end_date)
            df.loc[mask, "dep_end"] = dep_value
    return df

labeled = label_dep_end(labeled, dep_end)

# --- Step 6: (Optional) Convert Boolean Labels to Integer Values ---
labeled["dep_3day"] = labeled["dep_3day"].map({True: 1, False: 0})
labeled["dep_end"] = labeled["dep_end"].map({True: 1, False: 0})

# --- Step 7: Save Labeled Data to the '4_data_labeled' Folder ---
os.makedirs("4_data_labeled", exist_ok=True)
labeled.to_csv("4_data_labeled/location_labeled.csv", index=False)

print("✅ Labeled data saved to '4_data_labeled/location_labeled.csv'")

