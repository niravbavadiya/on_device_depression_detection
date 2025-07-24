import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical

# --- Config ---
BASE_DIR = Path(r"C:/Users/Nirav Bavadiya/Documents/study/Masters/Thesis/My work/data/3_data_pruned")
SENSOR_FILES = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
WINDOW_SIZE = 28
NUM_REORDER_CLASSES = 3  # Number of non-identity permutations
RATE_OF_REORDER = 0.5   # Fraction of training windows to reorder

# --- Generate random permutations ---
np.random.seed(42)
PERMUTATIONS = [np.random.permutation(WINDOW_SIZE) for _ in range(NUM_REORDER_CLASSES)]


def generate_windows(sensor_df, survey_df):
    windows, labels, pids = [], [], []
    sensor_df["date"] = pd.to_datetime(sensor_df["date"])
    survey_df["date"] = pd.to_datetime(survey_df["date"])

    for pid in survey_df["pid"].unique():
        svy_pid = survey_df[survey_df["pid"] == pid].sort_values("date")
        sen_pid = sensor_df[sensor_df["pid"] == pid]
        if svy_pid.empty or sen_pid.empty:
            continue

        for _, row in svy_pid.iterrows():
            label_date = row["date"]
            label = row["label"]
            date_range = pd.date_range(label_date - pd.Timedelta(days=WINDOW_SIZE - 1), label_date)
            placeholder = pd.DataFrame({"date": date_range})
            placeholder["pid"] = pid
            actual = sen_pid[(sen_pid["date"] >= date_range.min()) & (sen_pid["date"] <= date_range.max())]
            merged = pd.merge(placeholder, actual, on=["pid", "date"], how="left")
            features = merged.drop(columns=["pid", "date"])
            windows.append(features.values)
            labels.append(label)
            pids.append(pid)

    return np.array(windows), np.array(labels), np.array(pids)


# --- Load and encode all surveys ---
all_surveys = []
for dataset in BASE_DIR.iterdir():
    if not dataset.is_dir():
        continue
    svy_path = dataset / "SurveyData" / "dep_weekly.csv"
    if svy_path.exists():
        survey = pd.read_csv(svy_path)
        survey["date"] = pd.to_datetime(survey["date"])
        all_surveys.append(survey)

survey_all = pd.concat(all_surveys, ignore_index=True).dropna(subset=["dep"])
label_encoder = LabelEncoder()
survey_all["label"] = label_encoder.fit_transform(survey_all["dep"])
target_names = list(label_encoder.classes_)

# --- Collect data ---
X_all, y_all, reorder_all, pids_all = [], [], [], []
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

for dataset in BASE_DIR.iterdir():
    if not dataset.is_dir():
        continue
    print(f"\n📦 Processing dataset: {dataset.name}")

    merged = None
    for sensor_file in SENSOR_FILES:
        sensor_path = dataset / "FeatureData" / sensor_file
        if not sensor_path.exists():
            continue

        df = pd.read_csv(sensor_path)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        df.sort_values(["pid", "date"], inplace=True)
        features = df.columns.difference(["pid", "date"])
        df[features] = StandardScaler().fit_transform(df[features])
        df = df.rename(columns={col: f"{sensor_file[:-4]}_{col}" for col in features})
        merged = df if merged is None else pd.merge(merged, df, on=["pid", "date"], how="outer")

    if merged is None or merged.empty:
        print("⚠️ No sensor data found.")
        continue

    merged.fillna(0, inplace=True)
    survey = pd.read_csv(dataset / "SurveyData" / "dep_weekly.csv")
    survey["date"] = pd.to_datetime(survey["date"])
    survey = survey.dropna(subset=["dep"])
    survey["label"] = label_encoder.transform(survey["dep"])

    X, y, pids = generate_windows(merged, survey)
    if len(X) == 0:
        print("⚠️ No windows generated.")
        continue

    for train_idx, test_idx in splitter.split(X, y):
        for idx, is_train in zip(range(len(X)), [i in train_idx for i in range(len(X))]):
            x = X[idx]
            reorder_class = 0
            if is_train and np.random.rand() < RATE_OF_REORDER:
                reorder_class = np.random.randint(1, NUM_REORDER_CLASSES + 1)
                x = x[PERMUTATIONS[reorder_class - 1]]
            X_all.append(x)
            y_all.append(y[idx])
            reorder_all.append(reorder_class)
            pids_all.append(pids[idx])

# --- Final arrays ---
X_all = np.array(X_all)
y_all = to_categorical(np.array(y_all))
y_reorder = to_categorical(np.array(reorder_all), num_classes=NUM_REORDER_CLASSES + 1)
pids_all = np.array(pids_all)

# --- Save outputs ---
OUTPUT_DIR = BASE_DIR / "reorder_ready"
OUTPUT_DIR.mkdir(exist_ok=True)
np.save(OUTPUT_DIR / "X_windows.npy", X_all)
np.save(OUTPUT_DIR / "y_labels.npy", y_all)
np.save(OUTPUT_DIR / "y_reorder.npy", y_reorder)
np.save(OUTPUT_DIR / "pids.npy", pids_all)

print("✅ Reorder-ready data saved to:", OUTPUT_DIR)
