import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
import tensorflow as tf
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Conv1D, MaxPooling1D,
    LSTM, Dense, Dropout
)
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

# --- Run Directory & Logging ---
def get_next_folder_name(prefix):
    existing = [p for p in GRAPH_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    nums     = [int(p.name.split("_")[-1]) for p in existing
                if "_" in p.name and p.name.split("_")[-1].isdigit()]
    return GRAPH_DIR / f"{prefix}_{max(nums, default=0) + 1}"

RUN_DIR          = get_next_folder_name('run')
RUN_DIR.mkdir(parents=True, exist_ok=True)
log_file_path    = RUN_DIR / "run_output_log.txt"
sys.stdout       = open(log_file_path, "w", encoding="utf-8")

def get_next_filename(prefix, ext="png"):
    existing = list(RUN_DIR.glob(f"{prefix}_*.{ext}"))
    nums     = [int(p.stem.split("_")[-1]) for p in existing
                if p.stem.split("_")[-1].isdigit()]
    return RUN_DIR / f"{prefix}_{max(nums, default=0) + 1}.{ext}"

# --- Window Generator ---
def generate_fixed_windows(sensor_df, survey_df, window_size=WINDOW_SIZE):
    windows, labels, quality_log = [], [], []
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
            drange     = pd.date_range(label_date - pd.Timedelta(days=window_size-1),
                                       label_date)
            placeholder = pd.DataFrame({"date": drange})
            placeholder["pid"] = pid

            actual = sen_pid[(sen_pid["date"] >= drange.min()) &
                             (sen_pid["date"] <= drange.max())]
            merged = pd.merge(placeholder, actual, on=["pid","date"], how="left")
            feats  = merged.drop(columns=["pid","date"])
            missing_days = feats.isna().all(axis=1).sum()

            windows.append(feats.fillna(0).values)
            labels.append(label)
            quality_log.append({
                "pid": pid,
                "label_date": label_date,
                "missing_days": missing_days,
                "missing_ratio": round(missing_days/window_size, 3)
            })

    return np.array(windows), np.array(labels), pd.DataFrame(quality_log)

# --- Load & Encode Surveys ---
all_surveys = []
for ds in DATASETS:
    path = ds / "SurveyData" / "dep_weekly.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        all_surveys.append(df)

survey_all      = pd.concat(all_surveys, ignore_index=True).dropna(subset=["dep"])
label_encoder   = LabelEncoder()
survey_all["label"] = label_encoder.fit_transform(survey_all["dep"]) 
target_names    = [str(lbl) for lbl in label_encoder.classes_]

# --- Assemble Train & Test Sets ---
X_train_all, y_train_all = [], []
X_test_all,  y_test_all  = [], []

for ds in DATASETS:
    print(f"\n📦 Processing dataset: {ds.name}")
    merged = None

    # Merge & normalize sensors
    for sf in SENSOR_FILES:
        fp = ds / "FeatureData" / sf
        if not fp.exists():
            continue

        df = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["date"])
        df.fillna(0, inplace=True)
        df.sort_values(["pid","date"], inplace=True)

        feats = df.columns.difference(["pid","date"])
        #df[feats] = StandardScaler().fit_transform(df[feats])
        df.rename(columns={c: f"{sf[:-4]}_{c}" for c in feats}, inplace=True)

        merged = df if merged is None else pd.merge(
            merged, df, on=["pid","date"], how="outer"
        )

    if merged is None or merged.empty:
        print("⚠️ No sensor data found.")
        continue

    merged.to_csv(RUN_DIR / f"full_dataset_{ds.name}.csv", index=False)

    surv = pd.read_csv(ds / "SurveyData" / "dep_weekly.csv")
    surv["date"] = pd.to_datetime(surv["date"])
    surv = surv.dropna(subset=["dep"]) 
    surv["label"] = label_encoder.transform(surv["dep"]) 

    Xw, yw, qdf = generate_fixed_windows(merged, surv)
    qdf.to_csv(RUN_DIR / f"window_quality_{ds.name}.csv", index=False)

    if len(Xw) == 0:
        print("⚠️ No windows generated.")
        continue

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tr_idx, te_idx in splitter.split(Xw, yw):
        # 1. Extract raw train/test windows
        X_train, X_test = Xw[tr_idx], Xw[te_idx]
        y_train, y_test = yw[tr_idx], yw[te_idx]

        # 2. Scale feature-wise across all time steps
        n_windows, win_len, n_features = X_train.shape

        # Flatten windows into (n_windows * win_len, n_features)
        flat_train = X_train.reshape(-1, n_features)
        scaler = StandardScaler()
        flat_train_scaled = scaler.fit_transform(flat_train)
        X_train_scaled = flat_train_scaled.reshape(n_windows, win_len, n_features)

        # Transform test with the same scaler
        flat_test = X_test.reshape(-1, n_features)
        flat_test_scaled = scaler.transform(flat_test)
        X_test_scaled = flat_test_scaled.reshape(len(X_test), win_len, n_features)

        # 3. Accumulate scaled windows and labels
        X_train_all.extend(X_train_scaled)
        y_train_all.extend(y_train)
        X_test_all.extend(X_test_scaled)
        y_test_all.extend(y_test)

X_train = np.array(X_train_all)
X_test  = np.array(X_test_all)
y_train = to_categorical(np.array(y_train_all))
y_test  = to_categorical(np.array(y_test_all), num_classes=y_train.shape[1])

print(f"\nData shapes → X_train: {X_train.shape}, X_test: {X_test.shape}")

# --- Model Definition ---
def build_base_model(input_shape, num_classes):
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- Pruning Setup ---
batch_size = 4
epochs     = 50
steps_per_epoch = np.ceil(X_train.shape[0]/batch_size).astype(np.int32)
end_step        = steps_per_epoch * epochs

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.80,
        begin_step=0,
        end_step=end_step
    )
}

base_model       = build_base_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
pruned_model     = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

pruned_model.compile(
    optimizer=Adam(1e-3, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=4)
]

# --- Pruned Training ---
print("\n--- Training with Unstructured Pruning ---")
pruned_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=pruning_callbacks,
    verbose=1
)

# Strip pruning wrappers
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
final_model.compile(
    optimizer=Adam(1e-3, clipnorm=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
final_model.save(RUN_DIR / "pruned_model.keras")
print("✅ Pruning wrappers stripped and Keras model saved.")

# ----------------------- Quantization-Aware Training (QAT) -----------------------
# We'll convert the stripped (pruned) model into a quantization-aware model and fine-tune it.
# Notes:
#  - Some ops (e.g. LSTM) can limit ability to fully convert to integer-only TFLite. We try full integer first
#    and fall back to allowing SELECT_TF_OPS if necessary.
#  - QAT uses "fake quant" during training to simulate int8 behavior. We then use a representative dataset
#    when converting to TFLite to enable better post-training quantization.

print("\n--- Preparing Quantization-Aware Training (QAT) ---")
# ------------------ Selective QAT annotation (robust, multi-step fallback) ------------------
# We try to annotate Dense+Conv1D first, then fall back to Dense-only.
def make_annotator(layer_types_to_annotate):
    def apply_quantize_or_pass(layer):
        # Keep BatchNormalization in float to avoid custom QuantizeConfig
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return layer
        # Annotate only requested layer types
        for lt in layer_types_to_annotate:
            if isinstance(layer, lt):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
        # All other layers left as-is (float)
        return layer
    return apply_quantize_or_pass

quantize_attempts = [
    (tf.keras.layers.Dense, tf.keras.layers.Conv1D),  # attempt wider coverage first
    (tf.keras.layers.Dense,),                         # fallback: only Dense
]

last_exception = None
q_aware_model = None

for types_tuple in quantize_attempts:
    try:
        annotated_model = tf.keras.models.clone_model(
            final_model,
            clone_function=make_annotator(types_tuple),
        )

        # Convert annotations into a quantize-aware model
        with tfmot.quantization.keras.quantize_scope():
            q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

        print(f"✅ Quantization annotation succeeded with layer types: {types_tuple}")
        break

    except Exception as e:
        last_exception = e
        print(f"⚠️ Quantize attempt with {types_tuple} failed: {e}")

if q_aware_model is None:
    # Nothing worked — raise the last error so you can inspect it
    raise last_exception

# Compile the QAT model (fine-tune as before)
q_aware_model.compile(
    optimizer=Adam(1e-4),  # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Compile the QAT model (fine-tune as before)
q_aware_model.compile(
    optimizer=Adam(1e-4),  # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


q_aware_model.compile(
    optimizer=Adam(1e-4),  # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

qat_epochs = 10
qat_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3)
]

print("Starting QAT fine-tuning...")
q_aware_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=qat_epochs,
    batch_size=batch_size,
    callbacks=qat_callbacks,
    verbose=1
)

# save q-aware model
q_aware_model.save(RUN_DIR / "q_aware_model.keras")
print("✅ QAT model fine-tuned and saved.")

# --- TensorFlow Lite Runtime Conversion (LiteRT) from QAT model ---
print("\n--- Converting QAT model to TensorFlow Lite Runtime (LiteRT) ---")

# Representative dataset generator for better quantization
def representative_data_gen():
    # yield up to 100 samples from training set
    n = min(100, X_train.shape[0])
    for i in range(n):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]

# Try to get the most quantized TFLite possible; if LSTM prevents full integer quantization,
# fall back to allowing SELECT_TF_OPS (TF fallback kernels).
q_lite_path = RUN_DIR / "model_q_aware_litert.tflite"

# Attempt full TFLite builtins quantization first
try:
    converter = tflite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tflite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Request full integer ops
    converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS]
    # avoid lowering pass for dynamic shapes
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    with open(q_lite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ QAT full-builtins TFLite saved to: {q_lite_path}")
except Exception as e_full:
    print("⚠️ Full-builtins conversion failed (likely due to LSTM/unsupported ops). Falling back to SELECT_TF_OPS.")
    print(f"Error: {e_full}")

    converter = tflite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tflite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
        tflite.OpsSet.TFLITE_BUILTINS,
        tflite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    with open(q_lite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ QAT TFLite (with SELECT_TF_OPS) saved to: {q_lite_path}")


#-------------------------------------LiteRT eval------------------------------------------------------------------
from tensorflow.lite.python.interpreter import Interpreter as tflite_rt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the LiteRT model
interpreter = tflite_rt(model_path=str(q_lite_path))
interpreter.allocate_tensors()

# Get input/output details
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Collect predictions
lite_preds = []
for sample in X_test:
    input_data = sample.astype(np.float32)

    # Expand dims to match expected input shape
    interpreter.set_tensor(input_details['index'], np.expand_dims(input_data, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]
    lite_preds.append(output)

# Convert predictions to class indices
lite_preds = np.array(lite_preds)
lite_classes = np.argmax(lite_preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\n🧪 LiteRT (QAT) Model Evaluation:")
print(classification_report(true_classes, lite_classes, target_names=target_names))

# Optional: Confusion matrix
cm = confusion_matrix(true_classes, lite_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
ax_cm.set_title("LiteRT (QAT) Model Confusion Matrix")
plt.tight_layout()
fig_cm.savefig(get_next_filename("litert_qat_confusion_matrix"))
plt.close(fig_cm)


#-------------------------------------------------------------------------------------------------------

# --- Evaluation on Test Set ---
probs = q_aware_model.predict(X_test)
preds = np.argmax(probs, axis=1)
true  = np.argmax(y_test, axis=1)

print("\n🧪 QAT Keras Model Evaluation")
print(classification_report(true, preds, target_names=target_names))

train_acc = q_aware_model.evaluate(X_train, y_train, verbose=0)[1]
test_acc  = q_aware_model.evaluate(X_test,  y_test,  verbose=0)[1]
print(f"\n📊 Generalization Gap → Train: {train_acc:.3f}, Test: {test_acc:.3f}")

# --- Confusion Matrix & ROC Curves ---
cm   = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(cmap='Blues', ax=ax, colorbar=False)
ax.set_title("Confusion Matrix (QAT)")
fig.savefig(get_next_filename("confusion_matrix_qat"))
plt.close(fig)

# ROC Curve
fig = plt.figure()
if y_test.shape[1] == 2:
    fpr, tpr, _ = roc_curve(y_test[:,1], probs[:,1])
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
else:
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:,i], probs[:,i])
        plt.plot(fpr, tpr, label=f"{target_names[i]} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (QAT)")
plt.legend(loc='lower right')
plt.tight_layout()

fig.savefig(get_next_filename("roc_curve_qat"))
plt.close(fig)

print("\n--- Script Complete ---")
