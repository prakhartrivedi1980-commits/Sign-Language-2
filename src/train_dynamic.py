
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# =====================================
# CONFIGURATION
# =====================================

SEED = 42
EPOCHS = 80
BATCH_SIZE = 16

CLASSES = ["J", "Z", "OTHER"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "dynamic"
PROCESSED = DATA / "processed"

MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "dynamic"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "asl_jz_other_lstm.keras"
PREPROCESS_PATH = (
    MODELS / "asl_jz_other_preprocessing.npz"
)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================
# FEATURE TRANSFORMATION
# =====================================

def transform_sequences(X):
    X = np.asarray(X, dtype=np.float32).copy()

    if X.ndim != 3 or X.shape[1:] != (30, 66):
        raise ValueError(f"Invalid sequence shape: {X.shape}")

    # Keep 63 wrist-relative handshape features.
    # Convert wrist x/y to displacement from start.
    X[:, :, 63:65] -= X[:, :1, 63:65]

    # Wrist z is not reliable global depth.
    X[:, :, 65] = 0.0

    return X


# =====================================
# LOAD PUBLIC J/Z DATA
# =====================================

X_public = np.load(
    PROCESSED / "X_jz.npy"
).astype(np.float32)

y_public = np.load(
    PROCESSED / "y_jz.npy",
    allow_pickle=False
).astype(str)

# This split is for training/development.
# It is not a signer-independent evaluation.
idx = np.arange(len(X_public))

train_idx, holdout_idx = train_test_split(
    idx,
    test_size=0.20,
    stratify=y_public,
    random_state=SEED
)

X_parts = [X_public[train_idx]]
y_parts = [y_public[train_idx]]

# =====================================
# LOAD WEBCAM RECORDINGS
# =====================================

def load_webcam(split):
    features = []
    labels = []

    for label in CLASSES:
        folder = DATA / split / label

        for path in sorted(folder.rglob("*.npy")):
            sequence = np.load(
                path,
                allow_pickle=False
            ).astype(np.float32)

            if sequence.shape != (30, 66):
                print("Skipping invalid:", path)
                continue

            if not np.isfinite(sequence).all():
                print("Skipping non-finite:", path)
                continue

            features.append(sequence)
            labels.append(label)

    if not features:
        return (
            np.empty((0, 30, 66), dtype=np.float32),
            np.empty((0,), dtype=str)
        )

    return np.stack(features), np.array(labels)


X_webcam, y_webcam = load_webcam("train")

if len(X_webcam) == 0:
    raise ValueError("No webcam training sequences found")

X_parts.append(X_webcam)
y_parts.append(y_webcam)

X_train = np.concatenate(X_parts)
y_train_text = np.concatenate(y_parts)

# Validation must come from independent webcam sessions.
X_val, y_val_text = load_webcam("val")

if len(X_val) == 0:
    raise ValueError(
        "Collect webcam validation sequences first"
    )

if set(y_val_text) != set(CLASSES):
    raise ValueError(
        "Validation must contain J, Z and OTHER"
    )

# =====================================
# NORMALIZATION
# =====================================

X_train = transform_sequences(X_train)
X_val = transform_sequences(X_val)

mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1))

std[std < 1e-6] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

label_to_id = {
    label: i for i, label in enumerate(CLASSES)
}

y_train = np.array(
    [label_to_id[x] for x in y_train_text],
    dtype=np.int32
)

y_val = np.array(
    [label_to_id[x] for x in y_val_text],
    dtype=np.int32
)

print("\nTraining distribution:")
for label in CLASSES:
    print(label, np.sum(y_train_text == label))

print("\nValidation distribution:")
for label in CLASSES:
    print(label, np.sum(y_val_text == label))

# =====================================
# CLASS WEIGHTS
# =====================================

unique = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(unique, weights)
}

# =====================================
# LSTM ARCHITECTURE
# =====================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 66)),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.LSTM(32),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# TRAINING
# =====================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=5,
        factor=0.5,
        min_lr=1e-5
    )
]

print("\nTraining J/Z/OTHER LSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# =====================================
# VALIDATION METRICS
# =====================================

best = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

probabilities = best.predict(
    X_val,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

metrics = {
    "validation_accuracy": float(
        accuracy_score(y_val, predictions)
    ),
    "macro_precision": float(
        precision_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        )
    ),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "epochs_completed": len(history.history["loss"])
}

print("\nValidation metrics:")
print(json.dumps(metrics, indent=2))

report = classification_report(
    y_val,
    predictions,
    labels=range(3),
    target_names=CLASSES,
    zero_division=0
)

print("\nClassification Report:")
print(report)

cm = confusion_matrix(
    y_val,
    predictions,
    labels=range(3)
)

print("\nConfusion Matrix:")
print(cm)

# =====================================
# SAVE OUTPUTS
# =====================================

np.savez(
    PREPROCESS_PATH,
    mean=mean,
    std=std,
    classes=np.array(CLASSES, dtype=str),
    sequence_length=np.array(30),
    feature_version=np.array("wrist_xy_delta_v1")
)

with open(
    RESULTS / "jz_other_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=2)

with open(
    RESULTS / "jz_other_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

np.savetxt(
    RESULTS / "jz_other_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

print("\nTraining complete!")
print("Model:", MODEL_PATH)
print("Preprocessing:", PREPROCESS_PATH)
print("Original J/Z model preserved.")
