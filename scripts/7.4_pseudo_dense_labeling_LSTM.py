import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# --- Config ---
SENSOR_FILES = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
SENSOR_NAMES = ['location', 'screen', 'sleep', 'steps']
SEQUENCE_LEN = 5
ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned\INS-W-sample_1\FeatureData")
SURVEY_PATH = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned\INS-W-sample_1\SurveyData\dep_weekly.csv")

# --- Load survey and encode labels ---
survey = pd.read_csv(SURVEY_PATH)
survey["date"] = pd.to_datetime(survey["date"])
survey = survey.dropna(subset=["dep"])
label_encoder = LabelEncoder()
survey["label"] = label_encoder.fit_transform(survey["dep"])
target_names = [str(lbl) for lbl in label_encoder.classes_]

# --- Model Builder ---
def build_model(input_shape, output_dim):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        BatchNormalization(),
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# --- Fused output storage ---
participant_probs = defaultdict(list)
participant_labels = {}

# --- Loop per sensor ---
for sensor_file, sensor_name in zip(SENSOR_FILES, SENSOR_NAMES):
    print(f"\n▶ Processing: {sensor_name}")
    df = pd.read_csv(ROOT / sensor_file)
    df["date"] = pd.to_datetime(df["date"])
    df.fillna(0, inplace=True)
    feature_cols = df.columns.difference(['pid', 'date'])
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

    for held_pid in df["pid"].unique():
        df_train = df[df["pid"] != held_pid]
        df_test = df[df["pid"] == held_pid]
        svy_train = survey[survey["pid"] != held_pid]
        svy_test = survey[survey["pid"] == held_pid]
        if df_test.empty or svy_test.empty:
            continue

        def get_sequences(sensor_df, survey_df):
            X_seq, y_seq = [], []
            for pid in sensor_df["pid"].unique():
                part_sensor = sensor_df[sensor_df["pid"] == pid].sort_values("date")
                part_survey = survey_df[survey_df["pid"] == pid].sort_values("date")
                if part_sensor.empty or len(part_survey) < 1:
                    continue
                survey_dates = part_survey["date"].tolist()
                survey_labels = part_survey["label"].tolist()
                for i in range(len(survey_dates)):
                    start_date = part_sensor["date"].min() if i == 0 else survey_dates[i - 1] + pd.Timedelta(days=1)
                    end_date = survey_dates[i]
                    chunk = part_sensor[(part_sensor["date"] >= start_date) & (part_sensor["date"] <= end_date)]
                    if chunk.empty:
                        continue
                    features = chunk.sort_values(by="date").drop(columns=["pid", "date"]).values
                    for j in range(len(features) - SEQUENCE_LEN + 1):
                        window = features[j:j+SEQUENCE_LEN]
                        X_seq.append(window)
                        y_seq.append(survey_labels[i])
            return np.array(X_seq), np.array(y_seq)

        X_train, y_train = get_sequences(df_train, svy_train)
        X_test, y_test = get_sequences(df_test, svy_test)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test, num_classes=y_train_cat.shape[1])

        model = build_model(input_shape=(SEQUENCE_LEN, X_train.shape[2]), output_dim=y_train_cat.shape[1])
        callbacks = [
            EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="loss", patience=2)
        ]
        model.fit(X_train, y_train_cat, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)

        probs = model.predict(X_test, verbose=0)
        preds = np.argmax(probs, axis=1)
        held_pid_arr = df_test["pid"].unique()
        held_pid = held_pid_arr[0]
        participant_probs[held_pid].append(probs)
        participant_labels[held_pid] = y_test_cat
    print(f"✔ {sensor_name} done.")

# --- Fused Evaluation ---
print("\n🔗 Fused Model Evaluation (LOPOCV):")
fused_preds = []
fused_true = []
fused_all_probs = []

for pid in participant_probs:
    sensor_outputs = participant_probs[pid]
    avg_probs = np.mean(np.array(sensor_outputs), axis=0)
    preds = np.argmax(avg_probs, axis=1)
    true = np.argmax(participant_labels[pid], axis=1)
    fused_preds.extend(preds)
    fused_true.extend(true)
    fused_all_probs.append(avg_probs)

print(classification_report(fused_true, fused_preds, target_names=target_names))
cm = confusion_matrix(fused_true, fused_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
plt.title("LOPO – Fused Confusion Matrix")
plt.tight_layout()
plt.show()

# --- Fused ROC ---
fused_all_probs = np.concatenate(fused_all_probs)
fused_true_bin = to_categorical(fused_true, num_classes=len(target_names))

plt.figure()
if len(target_names) == 2:
    fpr, tpr, _ = roc_curve(fused_true_bin[:, 1], fused_all_probs[:, 1])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
else:
    for i in range(len(target_names)):
        fpr, tpr, _ = roc_curve(fused_true_bin[:, i], fused_all_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {target_names[i]} (AUC={auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LOPO – Fused ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
