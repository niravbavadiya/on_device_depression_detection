# merge all modality for each sample, do the labeling, train-test split then merge train/test

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- Config ---
BASE_DIR = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned")
DATASETS = [d for d in BASE_DIR.iterdir() if d.is_dir()]
SENSOR_FILES = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']

# --- Label encoding for survey ---
all_surveys = []
for dataset in DATASETS:
    survey_path = dataset / "SurveyData" / "dep_weekly.csv"
    if survey_path.exists():
        survey = pd.read_csv(survey_path)
        survey["date"] = pd.to_datetime(survey["date"])
        survey["dataset"] = dataset.name
        all_surveys.append(survey)

survey_all = pd.concat(all_surveys, ignore_index=True)
survey_all = survey_all.dropna(subset=["dep"])
label_encoder = LabelEncoder()
survey_all["label"] = label_encoder.fit_transform(survey_all["dep"])
target_names = [str(lbl) for lbl in label_encoder.classes_]

# --- Storage ---
X_train_all, y_train_all = [], []
X_test_all, y_test_all = [], []

# --- Process each dataset ---
for dataset in DATASETS:
    print(f"\n📦 Processing dataset: {dataset.name}")

    # Load and merge all sensor modalities
    merged = None
    for sensor_file in SENSOR_FILES:
        sensor_path = dataset / "FeatureData" / sensor_file
        if not sensor_path.exists():
            continue
        df = pd.read_csv(sensor_path)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        df = df.sort_values(["pid", "date"])
        feature_cols = df.columns.difference(["pid", "date"])
        df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])
        df = df.rename(columns={col: f"{sensor_file[:-4]}_{col}" for col in feature_cols})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["pid", "date"], how="outer")

    if merged is None or merged.empty:
        print(f"⚠️ No sensor data in {dataset.name}, skipping.")
        continue

    merged.fillna(0, inplace=True)

    # Load survey
    survey_path = dataset / "SurveyData" / "dep_weekly.csv"
    if not survey_path.exists():
        continue
    survey = pd.read_csv(survey_path)
    survey["date"] = pd.to_datetime(survey["date"])
    survey = survey.dropna(subset=["dep"])
    survey["label"] = label_encoder.transform(survey["dep"])

    # Generate windows
    X_windows, y_labels = [], []
    for pid in merged["pid"].unique():
        part_sensor = merged[merged["pid"] == pid].sort_values("date")
        part_survey = survey[survey["pid"] == pid].sort_values("date")
        if part_sensor.empty or len(part_survey) == 0:
            continue

        survey_dates = part_survey["date"].tolist()
        survey_labels = part_survey["label"].tolist()

        # First window
        first_window = part_sensor[part_sensor["date"] <= survey_dates[0]]
        if not first_window.empty:
            X_windows.append(first_window.drop(columns=["pid", "date"]).values)
            y_labels.append(survey_labels[0])

        # Remaining windows
        for i in range(1, len(survey_dates)):
            start_date = survey_dates[i - 1] + pd.Timedelta(days=1)
            end_date = survey_dates[i]
            window = part_sensor[(part_sensor["date"] >= start_date) & (part_sensor["date"] <= end_date)]
            if not window.empty:
                X_windows.append(window.drop(columns=["pid", "date"]).values)
                y_labels.append(survey_labels[i])

    if not X_windows:
        print(f"⚠️ No windows in {dataset.name}, skipping.")
        continue

    # Pad and split
    max_len = max(len(w) for w in X_windows)
    X_padded = pad_sequences(X_windows, maxlen=max_len, padding='post', dtype='float32')
    y_encoded = np.array(y_labels)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for train_idx, test_idx in splitter.split(X_padded, y_encoded):
        X_train_all.extend(X_padded[train_idx])
        y_train_all.extend(y_encoded[train_idx])
        X_test_all.extend(X_padded[test_idx])
        y_test_all.extend(y_encoded[test_idx])

# --- Final train/test sets ---
X_train = np.array(X_train_all)
y_train = to_categorical(np.array(y_train_all))
X_test = np.array(X_test_all)
y_test = to_categorical(np.array(y_test_all), num_classes=y_train.shape[1])

# --- Build and train model ---
def build_model(input_shape, output_dim):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        BatchNormalization(),
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=y_train.shape[1])
callbacks = [
    EarlyStopping(monitor="loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="loss", patience=3)
]
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=4, callbacks=callbacks, verbose=0)

# --- Evaluation ---
probs = model.predict(X_test)
preds = np.argmax(probs, axis=1)
true = np.argmax(y_test, axis=1)

print("\n🧪 Final Model Evaluation:")
print(classification_report(true, preds, target_names=target_names))
cm = confusion_matrix(true, preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# --- ROC Curve ---
plt.figure()
if y_test.shape[1] == 2:
    fpr, tpr, _ = roc_curve(y_test[:, 1], probs[:, 1])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
else:
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {target_names[i]} (AUC={auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
