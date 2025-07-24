# do the labeling and then train-test split for each modality and then merge train/test

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
# --- Paths ---
BASE_DIR = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned")
DATASETS = [d for d in BASE_DIR.iterdir() if d.is_dir()]
sensor_files = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
sensor_names = ['location', 'screen', 'sleep', 'steps']

# --- Load and merge survey data ---
all_surveys = []
for dataset in DATASETS:
    survey_path = dataset / "SurveyData" / "dep_weekly.csv"
    if survey_path.exists():
        survey = pd.read_csv(survey_path)
        survey["date"] = pd.to_datetime(survey["date"])
        survey["dataset"] = dataset.name
        all_surveys.append(survey)

survey = pd.concat(all_surveys, ignore_index=True)
survey = survey.dropna(subset=["dep"])
label_encoder = LabelEncoder()
survey["label"] = label_encoder.fit_transform(survey["dep"])
target_names = [str(lbl) for lbl in label_encoder.classes_]


# Store for fusion
X_fused_test_probs = []
y_fused_test = None
models = {}

# Model builder
def build_model(input_shape, output_dim):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        BatchNormalization(),
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),  # First LSTM layer
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),  # Second LSTM layer
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Per-sensor training
for sensor_file, sensor_name in zip(sensor_files, sensor_names):
    print(f"\n🔹 Processing sensor: {sensor_name}")

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for dataset in DATASETS:
        sensor_path = dataset / "FeatureData" / sensor_file
        survey_path = dataset / "SurveyData" / "dep_weekly.csv"
        if not sensor_path.exists() or not survey_path.exists():
            continue

        df = pd.read_csv(sensor_path)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        feature_cols = df.columns.difference(['pid', 'date'])
        df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

        survey = pd.read_csv(survey_path)
        survey["date"] = pd.to_datetime(survey["date"])
        survey = survey.dropna(subset=["dep"])
        survey["label"] = label_encoder.transform(survey["dep"])

        X_windows, y_labels = [], []
        for pid in df["pid"].unique():
            part_sensor = df[df["pid"] == pid].sort_values("date")
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
            continue

        # Pad and split windows for this dataset
        max_len = max(len(w) for w in X_windows)
        X_padded = pad_sequences(X_windows, maxlen=max_len, padding='post', dtype='float32')
        y_encoded = np.array(y_labels)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for train_idx, test_idx in splitter.split(X_padded, y_encoded):
            X_train_all.extend(X_padded[train_idx])
            y_train_all.extend(y_encoded[train_idx])
            X_test_all.extend(X_padded[test_idx])
            y_test_all.extend(y_encoded[test_idx])

    if not X_train_all:
        print(f"⚠️ No data for {sensor_name}, skipping.")
        continue

    # Convert to arrays
    X_train = np.array(X_train_all)
    y_train = to_categorical(np.array(y_train_all))
    X_test = np.array(X_test_all)
    y_test = to_categorical(np.array(y_test_all), num_classes=y_train.shape[1])
    y_fused_test = y_test  # for fusion

    # Train model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=y_train.shape[1])
    callbacks = [
        EarlyStopping(monitor="loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="loss", patience=3)
    ]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=4, callbacks=callbacks, verbose=0)
    models[sensor_name] = model

    # Evaluate and store predictions
    probs = model.predict(X_test)
    X_fused_test_probs.append(probs)

# --- FUSION ---
print("\n🧪 Fused Model Evaluation:")
fused_probs = np.mean(np.array(X_fused_test_probs), axis=0)
fused_preds = np.argmax(fused_probs, axis=1)
fused_true = np.argmax(y_fused_test, axis=1)

print(classification_report(fused_true, fused_preds, target_names=target_names))
cm = confusion_matrix(fused_true, fused_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
plt.title("Fused Model Confusion Matrix")
plt.tight_layout()
plt.show()

# Fused ROC
plt.figure()
if y_fused_test.shape[1] == 2:
    fpr, tpr, _ = roc_curve(y_fused_test[:, 1], fused_probs[:, 1])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Fused Model (AUC={auc_score:.2f})")
else:
    for i in range(y_fused_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_fused_test[:, i], fused_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fused ROC Curve")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
