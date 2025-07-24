import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import sys
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- Config ---
BASE_DIR    = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\data\3_data_pruned")
GRAPH_ROOT  = Path(r"C:\Users\Nirav Bavadiya\Documents\study\Masters\Thesis\My work\graphs")
SCRIPT_NAME = Path(sys.argv[0]).stem if len(sys.argv) > 0 else "main"
GRAPH_DIR   = GRAPH_ROOT / SCRIPT_NAME
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_FILES = ['location.csv', 'screen.csv', 'sleep.csv', 'steps.csv']
WINDOW_SIZE  = 28
DATASETS     = [d for d in BASE_DIR.iterdir() if d.is_dir()]

def get_next_filename(prefix, ext="png"):
    existing = list(GRAPH_DIR.glob(f"{prefix}_*.{ext}"))
    nums = [int(p.stem.split("_")[-1]) for p in existing 
            if p.stem.split("_")[-1].isdigit()]
    next_num = max(nums, default=0) + 1
    return GRAPH_DIR / f"{prefix}_{next_num}.{ext}"

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
            label      = row["label"]
            date_range = pd.date_range(
                label_date - pd.Timedelta(days=window_size - 1),
                label_date
            )
            placeholder = pd.DataFrame({"date": date_range})
            placeholder["pid"] = pid
            actual = sen_pid[
                (sen_pid["date"] >= date_range.min()) &
                (sen_pid["date"] <= date_range.max())
            ]
            merged = pd.merge(
                placeholder, actual,
                on=["pid", "date"], how="left"
            )
            features = merged.drop(columns=["pid", "date"]).fillna(0)
            windows.append(features.values)
            labels.append(label)

    return np.array(windows), np.array(labels)

# --- Load & encode surveys ---
all_surveys = []
for dataset in DATASETS:
    svy_path = dataset / "SurveyData" / "dep_weekly.csv"
    if svy_path.exists():
        df = pd.read_csv(svy_path)
        df["date"] = pd.to_datetime(df["date"])
        all_surveys.append(df)

survey_all = pd.concat(all_surveys, ignore_index=True)\
                 .dropna(subset=["dep"])
label_encoder = LabelEncoder()
survey_all["label"] = label_encoder.fit_transform(survey_all["dep"])
target_names = [str(lbl) for lbl in label_encoder.classes_]

# --- Aggregate windows across all datasets ---
X_all, y_all = [], []

for dataset in DATASETS:
    merged = None
    for sensor_file in SENSOR_FILES:
        sensor_path = dataset / "FeatureData" / sensor_file
        if not sensor_path.exists():
            continue
        df = pd.read_csv(sensor_path)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        df.sort_values(["pid", "date"], inplace=True)
        fcols = df.columns.difference(["pid", "date"])
        df[fcols] = StandardScaler().fit_transform(df[fcols])
        df.rename(columns={c: f"{sensor_file[:-4]}_{c}" for c in fcols},
                  inplace=True)

        merged = df if merged is None else pd.merge(
            merged, df, on=["pid", "date"], how="outer"
        )

    if merged is None or merged.empty:
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

X = np.array(X_all)                         # shape: (N, 28, F)
y_labels = np.array(y_all)                  # shape: (N,)
y = to_categorical(y_labels)                # shape: (N, num_classes)

# --- Model definition ---
def build_model(input_shape, output_dim):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        LSTM(64, return_sequences=True, dropout=0.3),
        LSTM(32, dropout=0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(0.001, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- Stratified K-Fold CV ---
n_splits = 5
skf = StratifiedKFold(
    n_splits=n_splits, shuffle=True, random_state=42
)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y_labels), start=1):

    print(f"\n▶️ Fold {fold}/{n_splits}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # build fresh model
    model = build_model(
        (X_tr.shape[1], X_tr.shape[2]),
        y_tr.shape[1]
    )

    # train
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=4,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=7,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", patience=3
            )
        ],
        verbose=1
    )

    # evaluate
    probs = model.predict(X_val)
    preds = np.argmax(probs, axis=1)
    true  = np.argmax(y_val, axis=1)

    report = classification_report(
        true, preds, target_names=target_names,
        output_dict=True
    )
    acc_weighted = report["weighted avg"]["precision"]
    f1_weighted  = report["weighted avg"]["f1-score"]

    # ROC AUC (binary only)
    if y_val.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_val[:, 1], probs[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = None

    # save fold metrics
    fold_results.append({
        "fold": fold,
        "accuracy": report["accuracy"],
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc
    })

    # confusion matrix
    cm = confusion_matrix(true, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=target_names
    )
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"Conf Matrix Fold {fold}")
    plt.tight_layout()
    fig_cm.savefig(get_next_filename(f"confusion_matrix_fold{fold}"))
    plt.close(fig_cm)

    # ROC curve
    if roc_auc is not None:
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title(f"ROC Fold {fold}")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
        ax_roc.legend(loc='lower right')
        plt.tight_layout()
        fig_roc.savefig(get_next_filename(f"roc_curve_fold{fold}"))
        plt.close(fig_roc)

# --- Summarize CV results ---
import pandas as pd
df_results = pd.DataFrame(fold_results).set_index('fold')
print("\n🔍 Cross-Validation Results per Fold:")
print(df_results)
print("\n✨ Mean Scores Across Folds:")
print(df_results.mean())
