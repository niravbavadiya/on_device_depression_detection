# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import sys

# 🧠 TensorFlow model components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers  # L2 regularization

# --- Config Paths ---
BASE_DIR    = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned")
GRAPH_ROOT  = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\graphs")
SCRIPT_NAME = Path(sys.argv[0]).stem if len(sys.argv) > 0 else "main"
GRAPH_DIR   = GRAPH_ROOT / SCRIPT_NAME
MODEL_DIR   = GRAPH_DIR / 'model'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# --- Global Settings ---
SENSOR_FILES = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
WINDOW_SIZE  = 28
DATASETS = [d for d in BASE_DIR.iterdir() if d.is_dir()]

# --- Utility to save graphs ---
def get_next_folder_name(prefix):

    existing_dirs = [p for p in GRAPH_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    nums = [
        int(p.name.split("_")[-1])
        for p in existing_dirs
        if "_" in p.name and p.name.split("_")[-1].isdigit()
    ]
    next_num = max(nums, default=0) + 1
    return GRAPH_DIR / f"{prefix}_{next_num}"

RUN_DIR = get_next_folder_name('run')
RUN_DIR.mkdir(parents=True, exist_ok=True)

log_file_path = RUN_DIR / "run_output_log.txt"
sys.stdout = open(log_file_path, "w", encoding="utf-8")

def get_next_filename(prefix, ext="png"):
    existing = list(GRAPH_DIR.glob(f"{prefix}_*.{ext}"))
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    next_num = max(nums, default=0) + 1
    return RUN_DIR / f"{prefix}_{next_num}.{ext}"

# --- Window Generator: extract 28-day sensor window per label ---
def generate_fixed_windows(sensor_df, survey_df, window_size=28):
    windows, labels = [], []
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

            # Build placeholder with 28 days leading up to survey
            date_range = pd.date_range(label_date - pd.Timedelta(days=window_size - 1), label_date)
            placeholder = pd.DataFrame({"date": date_range})
            placeholder["pid"] = pid

            actual = sen_pid[(sen_pid["date"] >= date_range.min()) & (sen_pid["date"] <= date_range.max())]
            merged = pd.merge(placeholder, actual, on=["pid", "date"], how="left")

            features = merged.drop(columns=["pid", "date"]).fillna(0)
            windows.append(features.values)
            labels.append(label)

    return np.array(windows), np.array(labels)

# --- Survey Label Encoding ---
all_surveys = []


for dataset in DATASETS:
    svy_path = dataset / "SurveyData" / "dep_weekly.csv"
    if svy_path.exists():
        survey = pd.read_csv(svy_path)
        survey["date"] = pd.to_datetime(survey["date"])
        all_surveys.append(survey)

survey_all = pd.concat(all_surveys, ignore_index=True).dropna(subset=["dep"])
label_encoder = LabelEncoder()
survey_all["label"] = label_encoder.fit_transform(survey_all["dep"])
target_names = [str(lbl) for lbl in label_encoder.classes_]

# --- Build Input Features & Labels ---
X_all, y_all = [], []

for dataset in DATASETS:
    print(f"\n📦 Processing: {dataset.name}")
    merged = None

    for sensor_file in SENSOR_FILES:
        sensor_path = dataset / "FeatureData" / sensor_file
        if not sensor_path.exists(): continue

        df = pd.read_csv(sensor_path)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        df.sort_values(["pid", "date"], inplace=True)
        fcols = df.columns.difference(["pid", "date"])

        # 🚀 Normalize features = helps generalization
        df[fcols] = StandardScaler().fit_transform(df[fcols])

        # Rename feature columns to include source prefix
        df = df.rename(columns={col: f"{sensor_file[:-4]}_{col}" for col in fcols})
        merged = df if merged is None else pd.merge(merged, df, on=["pid", "date"], how="outer")

    if merged is None or merged.empty:
        print("⚠️ No sensor data found.")
        continue
    merged.fillna(0, inplace=True)

    svy_path = dataset / "SurveyData" / "dep_weekly.csv"
    survey = pd.read_csv(svy_path)
    survey["date"] = pd.to_datetime(survey["date"])
    survey = survey.dropna(subset=["dep"])
    survey["label"] = label_encoder.transform(survey["dep"])

    X_win, y_lbl = generate_fixed_windows(merged, survey)
    X_all.extend(X_win)
    y_all.extend(y_lbl)

# --- Final arrays ---
X = np.array(X_all)
y = to_categorical(np.array(y_all))
y_labels = np.argmax(y, axis=1)

# --- Train–Test Split (single shuffle) ---
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in splitter.split(X, y_labels):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# --- Model: CNN + LSTM + Dropout + Regularization ---
def build_model(input_shape, output_dim):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),

        # 🧠 Dropout improves generalization
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        LSTM(32, dropout=0.3, recurrent_dropout=0.2),

        # 🌱 L2 regularization to control weight magnitude
        Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.0001)),
        Dropout(0.3),

        Dense(output_dim, activation="softmax")
    ])

    # 🧠 Clipnorm controls gradient explosion
    model.compile(optimizer=Adam(0.001, clipnorm=1.0),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --- Training ---
model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
callbacks = [
    # 💡 Validation-based early stopping prevents memorization
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),

    # 📉 Reduce learning rate if validation stalls
    ReduceLROnPlateau(monitor="val_loss", patience=3)
]

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50, batch_size=4,
          callbacks=callbacks, verbose=1)

model.save(RUN_DIR / "trained_model_final.keras")

# --- Evaluation ---
probs = model.predict(X_test)
preds = np.argmax(probs, axis=1)
true  = np.argmax(y_test, axis=1)

print("\n🧪 Final Evaluation:")
print(classification_report(true, preds, target_names=target_names))

# 🎯 Generalization gap display
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc  = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\n📊 Generalization Gap: Train Acc = {train_acc:.3f} → Test Acc = {test_acc:.3f}")

# --- Confusion Matrix ---
cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
ax_cm.set_title("Confusion Matrix")
plt.tight_layout()
fig_cm.savefig(get_next_filename("confusion_matrix"))
plt.close(fig_cm)

# --- ROC Curve ---
fig_roc = plt.figure()

if y_test.shape[1] == 2:
    # 🧠 Binary classification ROC curve
    fpr, tpr, _ = roc_curve(y_test[:, 1], probs[:, 1])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
else:
    # 📊 Multiclass ROC: one curve per class
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {target_names[i]} (AUC={auc_score:.2f})")

# 🎨 Final formatting
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.tight_layout()

# 📁 Save plot
fig_roc.savefig(get_next_filename("roc_curve"))
plt.close(fig_roc)
