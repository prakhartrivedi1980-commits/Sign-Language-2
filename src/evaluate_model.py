
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "dataset" / "asl_landmarks_full.csv"

MODEL_PATH = ROOT / "models" / "asl_24_improved.keras"

PREPROCESS_PATH = (
    ROOT / "models" /
    "asl_24_improved_preprocessing.npz"
)

OUTPUT_DIR = ROOT / "results" / "improved_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load preprocessing
with np.load(PREPROCESS_PATH, allow_pickle=False) as data:
    mean = data["mean"]
    std = data["std"]
    classes = data["classes"].astype(str)

feature_cols = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

# Load dataset
df = pd.read_csv(DATA_PATH)

df = df[df["label"].isin(classes)].copy()

df = df.dropna(subset=["label"] + feature_cols)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols)

# Match the improved training script's Kaggle split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

X_test = test_df[feature_cols].to_numpy(
    dtype=np.float32
)

label_to_id = {
    label: i
    for i, label in enumerate(classes)
}

y_test = test_df["label"].map(
    label_to_id
).to_numpy(dtype=np.int32)

# Normalize with saved training parameters
X_test = (X_test - mean) / std

# Load trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Predictions
probabilities = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)

print("\nClassification Report:")
print(report)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(classes)))
)

# Save confusion matrix image
fig, ax = plt.subplots(figsize=(14, 12))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    values_format="d"
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200
)
plt.close()

# Save per-class metrics
report_dict = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "test_samples": int(len(y_test))
}

with open(
    OUTPUT_DIR / "evaluation_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nEvaluation files saved in:", OUTPUT_DIR)