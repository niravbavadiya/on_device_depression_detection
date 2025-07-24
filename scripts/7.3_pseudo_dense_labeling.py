import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# --- Config ---
sensor_files = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
sensor_names = ['location', 'screen', 'sleep', 'steps']
ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned\INS-W-sample_1\FeatureData")
SURVEY_PATH = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned\INS-W-sample_1\SurveyData\dep_weekly.csv")
# Load and encode survey data
survey = pd.read_csv(SURVEY_PATH)
survey["date"] = pd.to_datetime(survey["date"])
survey = survey.dropna(subset=["dep"]) 
label_encoder = LabelEncoder()
survey["label"] = label_encoder.fit_transform(survey["dep"])
target_names = [str(lbl) for lbl in label_encoder.classes_]
# CNN-friendly model builder (1-layer MLP here)
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(output_dim, activation="softmax")
    ]) 
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
# Store models and results
X_fused_test_probs = []
y_fused_test = None
# Main loop over sensor files
for sensor_file, sensor_name in zip(sensor_files, sensor_names):
    print(f"\n🔹 Processing sensor: {sensor_name}")
    df = pd.read_csv(ROOT / sensor_file)
    df["date"] = pd.to_datetime(df["date"])
    df.fillna(0, inplace=True)
    feature_cols = df.columns.difference(["pid", "date"])
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])
    # Pseudo-dense labeling logic
    X_rows, y_labels = [], []
    for pid in df["pid"].unique():
        part_sensor = df[df["pid"] == pid].sort_values("date")
        part_survey = survey[survey["pid"] == pid].sort_values("date")
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
            features = chunk.drop(columns=["pid", "date"]).values
            X_rows.extend(features)
            y_labels.extend([survey_labels[i]] * len(features))
    print(len(X_rows), len(y_labels))
    if not X_rows:
        print(f"⚠️ Skipping {sensor_name} — no labeled rows.")
        continue
    X = np.array(X_rows)
    y = np.array(y_labels)
    y_cat = to_categorical(y)
    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]
        y_fused_test = y_test  # for fusion comparison
    # Build + train model
    model = build_model(input_dim=X.shape[1], output_dim=y_cat.shape[1])
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
                 ReduceLROnPlateau(patience=2)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=50, batch_size=32, callbacks=callbacks)
    # Evaluate and plot
    probs = model.predict(X_test)
    X_fused_test_probs.append(probs)
    #preds = np.argmax(probs, axis=1)
    #true = np.argmax(y_test, axis=1)
    #print(f"\n📊 Report for {sensor_name}:")
    #print(classification_report(true, preds, target_names=target_names))
    #cm = confusion_matrix(true, preds)
    #ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
    #plt.title(f"{sensor_name.capitalize()} Confusion Matrix")
    #plt.tight_layout()
    #plt.show()
    ## ROC curve
    #if y_test.shape[1] == 2:
    #    fpr, tpr, _ = roc_curve(y_test[:, 1], probs[:, 1])
    #    auc_score = auc(fpr, tpr)
    #    plt.plot(fpr, tpr, label=f"{sensor_name} (AUC = {auc_score:.2f})")
    #else:
    #    for i in range(y_test.shape[1]):
    #        fpr, tpr, _ = roc_curve(y_test[:, i], probs[:, i])
    #        auc_score = auc(fpr, tpr)
    #        plt.plot(fpr, tpr, label=f"{sensor_name} – Class {i} (AUC = {auc_score:.2f})")
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.title(f"ROC – {sensor_name.capitalize()}")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
# --- Late Fusion ---
print("\n🧪 Fused Model Evaluation")
fused_probs = np.mean(np.array(X_fused_test_probs), axis=0)
print(fused_probs)
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
    plt.plot(fpr, tpr, label=f"Fused (AUC = {auc_score:.2f})")
else:
    for i in range(y_fused_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_fused_test[:, i], fused_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Fused ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
