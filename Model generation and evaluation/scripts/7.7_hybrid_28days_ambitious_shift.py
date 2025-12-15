import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import sys
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.layers import TimeDistributed, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- Config ---
BASE_DIR = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned")
GRAPH_ROOT = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\graphs")
SCRIPT_NAME = Path(sys.argv[0]).stem if len(sys.argv) > 0 else "main"
GRAPH_DIR = GRAPH_ROOT / SCRIPT_NAME
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_FILES = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
WINDOW_SIZE = 28
DATASETS = [d for d in BASE_DIR.iterdir() if d.is_dir()]

def get_next_filename(prefix, ext="png"):
    existing = list(GRAPH_DIR.glob(f"{prefix}_*.{ext}"))
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    next_num = max(nums, default=0) + 1
    return GRAPH_DIR / f"{prefix}_{next_num}.{ext}"

# --- Cumulative Window Builder ---
def build_cumulative_sequences(sensor_df, survey_df, window_size=28):
    sequences, labels = [], []
    sensor_df["date"] = pd.to_datetime(sensor_df["date"])
    survey_df["date"] = pd.to_datetime(survey_df["date"])

    for pid in survey_df["pid"].unique():
        svy_pid = survey_df[survey_df["pid"] == pid].sort_values("date")
        sen_pid = sensor_df[sensor_df["pid"] == pid]
        if svy_pid.empty or sen_pid.empty:
            continue

        history_windows = []

        for _, row in svy_pid.iterrows():
            label_date = row["date"]
            label = row["label"]
            date_range = pd.date_range(label_date - pd.Timedelta(days=window_size - 1), label_date)
            placeholder = pd.DataFrame({"date": date_range})
            placeholder["pid"] = pid
            actual = sen_pid[(sen_pid["date"] >= date_range.min()) & (sen_pid["date"] <= date_range.max())]
            merged = pd.merge(placeholder, actual, on=["pid", "date"], how="left")
            features = merged.drop(columns=["pid", "date"]).fillna(0).values

            history_windows.append(features)
            sequence = np.stack(history_windows, axis=0)  # shape: (t, 28, F)
            sequences.append(sequence)
            labels.append(label)

    return sequences, labels

# --- Load and encode survey labels ---
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

# --- Dataset loop with cumulative sequences ---
X_all, y_all = [], []
for dataset in DATASETS:
    print(f"\n📦 Processing dataset: {dataset.name}")
    merged = None

    for sensor_file in SENSOR_FILES:
        sensor_path = dataset / "FeatureData" / sensor_file
        if not sensor_path.exists(): continue

        df = pd.read_csv(sensor_path)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        df.sort_values(["pid", "date"], inplace=True)
        fcols = df.columns.difference(["pid", "date"])
        df[fcols] = StandardScaler().fit_transform(df[fcols])
        df = df.rename(columns={col: f"{sensor_file[:-4]}_{col}" for col in fcols})
        merged = df if merged is None else pd.merge(merged, df, on=["pid", "date"], how="outer")

    if merged is None or merged.empty:
        print("⚠️ Skipping empty dataset.")
        continue
    merged.fillna(0, inplace=True)

    svy_path = dataset / "SurveyData" / "dep_weekly.csv"
    survey = pd.read_csv(svy_path)
    survey["date"] = pd.to_datetime(survey["date"])
    survey = survey.dropna(subset=["dep"])
    survey["label"] = label_encoder.transform(survey["dep"])

    X_seq, y_seq = build_cumulative_sequences(merged, survey)
    X_all.extend(X_seq)
    y_all.extend(y_seq)

# --- Pad sequences to same length
max_len = max(len(seq) for seq in X_all)
num_features = X_all[0].shape[2]
X_padded = np.zeros((len(X_all), max_len, WINDOW_SIZE, num_features))

for i, seq in enumerate(X_all):
    X_padded[i, -len(seq):] = seq  # pad at beginning

y_encoded = to_categorical(np.array(y_all))

# --- Train/Test Split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in splitter.split(X_padded, y_all):
    X_train = X_padded[train_idx]
    y_train = y_encoded[train_idx]
    X_test = X_padded[test_idx]
    y_test = y_encoded[test_idx]

# --- Build cumulative model


def build_cumulative_model(input_shape, output_dim):
    seq_len, win_len, feat_dim = input_shape
    input_layer = Input(shape=(seq_len, win_len, feat_dim))

    # TimeDistributed CNN–LSTM per window
    td_layer = TimeDistributed(Sequential([
        Masking(mask_value=0.0, input_shape=(win_len, feat_dim)),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        LSTM(32, return_sequences=False, dropout=0.3)
    ]))(input_layer)

    # Two stacked LSTMs across windows with return_sequences
    final_seq = LSTM(64, return_sequences=True, dropout=0.3)(td_layer) # 2: added one more lstm layer, 3: Increase LSTM Units, return_sequences=True+ GlobalAveragePooling1D()
    final_seq = LSTM(32, dropout=0.3)(final_seq) 
    
    # Final classifier
    output_layer = Dense(output_dim, activation='softmax')(final_seq)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_cumulative_model((max_len, WINDOW_SIZE, num_features), y_train.shape[1])
callbacks = [
    EarlyStopping(monitor="loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="loss", patience=3)
]
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=4, callbacks=callbacks, verbose=0)

# --- Evaluation
probs = model.predict(X_test)
preds = np.argmax(probs, axis=1)
true = np.argmax(y_test, axis=1)

print("\n🧪 Evaluation Report:")
print(classification_report(true, preds, target_names=target_names))

# --- Confusion Matrix
cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
ax_cm.set_title("Confusion Matrix")
plt.tight_layout()
fig_cm.savefig(get_next_filename("confusion_matrix"))
plt.close(fig_cm)

# --- ROC Curve
fig_roc = plt.figure()

if y_test.shape[1] == 2:
    # Binary classification ROC
    fpr, tpr, _ = roc_curve(y_test[:, 1], probs[:, 1])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
else:
    # Multiclass ROC: one curve per class
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {target_names[i]} (AUC={auc_score:.2f})")

# Final formatting
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
fig_roc.savefig(get_next_filename("roc_curve"))
plt.close(fig_roc)
