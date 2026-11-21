"""ASL 24-class full training and versioned fine-tuning.

Examples:
  python src/train_model.py --mode finetune
  python src/train_model.py --mode full
  python src/train_model.py --mode finetune --base-model asl_24_finetuned_candidate.keras
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
FEATURES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
SEED = 42


def clean(df, label_column="label"):
    if label_column not in df.columns:
        raise ValueError(f"Missing {label_column} column")
    df = df.copy()
    df["label"] = df[label_column].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(LETTERS)].copy()
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing landmark columns: {sorted(missing)}")
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return df


def split_kaggle(df):
    # Development split. This must match the base model's training partition
    # to avoid replaying test samples; otherwise use saved split manifests.
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )
    return train, val, test


def load_webcam():
    folder = DATA / "webcam_improvement"
    parts = []
    if folder.exists():
        for path in sorted(folder.glob("train*.csv")):
            part = clean(pd.read_csv(path))
            print(f"Webcam {path.name}: {len(part)} usable samples")
            if not part.empty:
                parts.append(part)
    correction_path = DATA / "corrections" / "confirmed_corrections.csv"
    if correction_path.exists():
        corrections = clean(pd.read_csv(correction_path), "correct_label")
        print(f"Human corrections: {len(corrections)} usable samples")
        if not corrections.empty:
            parts.append(corrections)
    if not parts:
        return pd.DataFrame(columns=["label"] + FEATURES)
    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(subset=["label"] + FEATURES)


def arrays(df):
    labels = {letter: i for i, letter in enumerate(LETTERS)}
    return (
        df[FEATURES].to_numpy(dtype=np.float32),
        df["label"].map(labels).to_numpy(dtype=np.int32),
    )


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(24, activation="softmax"),
    ])


def main():
    parser = argparse.ArgumentParser(description="ASL full training or safe fine-tuning")
    parser.add_argument("--mode", choices=["full", "finetune"], default="finetune")
    parser.add_argument("--base-model", default="asl_24_finetuned_candidate.keras",
                        help="Existing model filename inside models/ (finetune only)")
    parser.add_argument("--replay-per-class", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if args.replay_per_class < 1:
        parser.error("--replay-per-class must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if Path(args.base_model).name != args.base_model or not args.base_model.endswith(".keras"):
        parser.error("--base-model must be a .keras filename inside models/")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    kaggle = clean(pd.read_csv(DATA / "asl_landmarks_full.csv"))
    train_k, val_k, test_k = split_kaggle(kaggle)
    print(f"Kaggle split: train={len(train_k)}, val={len(val_k)}, test={len(test_k)}")

    if args.mode == "finetune":
        base_path = MODELS / args.base_model
        preprocess_path = MODELS / f"{base_path.stem}_preprocessing.npz"
        if not base_path.is_file() or not preprocess_path.is_file():
            raise FileNotFoundError(f"Need both {base_path} and {preprocess_path}")
        with np.load(preprocess_path, allow_pickle=False) as saved:
            mean = saved["mean"].astype(np.float32)
            std = saved["std"].astype(np.float32)
            saved_classes = saved["classes"].astype(str).tolist()
        if saved_classes != LETTERS:
            raise ValueError("Base model class order differs from the 24 ASL letters")
        if mean.shape != (63,) or std.shape != (63,):
            raise ValueError("Base preprocessing must have 63 mean/std values")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Base preprocessing contains non-finite values")

        webcam = load_webcam()
        if webcam.empty:
            raise ValueError("No webcam training data or corrections found")
        min_count = train_k["label"].value_counts().min()
        replay = train_k.groupby("label", group_keys=False).sample(
            n=min(args.replay_per_class, int(min_count)), random_state=SEED
        )
        train_df = pd.concat([replay, webcam], ignore_index=True)
        train_df = train_df.drop_duplicates(subset=["label"] + FEATURES)
        model = tf.keras.models.load_model(base_path, compile=False)
        if model.output_shape[-1] != 24 or model.input_shape[-1] != 63:
            raise ValueError("Base model architecture is incompatible")
        learning_rate = 1e-4
        epochs = args.epochs or 20
        batch_size = 32
        print(f"Starting from: {base_path.name}")
        print(f"Replay samples: {len(replay)}; webcam/corrections: {len(webcam)}")
    else:
        train_df = train_k.drop_duplicates(subset=["label"] + FEATURES)
        raw = train_df[FEATURES].to_numpy(dtype=np.float32)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        model = build_model()
        learning_rate = 1e-3
        epochs = args.epochs or 100
        batch_size = 64

    std = np.where(std < 1e-6, 1.0, std)
    X_train, y_train = arrays(train_df)
    X_val, y_val = arrays(val_k)
    X_test, y_test = arrays(test_k)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print("\nTraining samples per class:")
    print(train_df["label"].value_counts().sort_index().to_string())
    unique = np.unique(y_train)
    if len(unique) != 24:
        raise ValueError("Training data must contain all 24 classes")
    weights = compute_class_weight("balanced", classes=unique, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(unique, weights)}

    # Unique filenames protect every previous candidate and its preprocessing.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"asl_24_{args.mode}_{timestamp}"
    candidate_path = MODELS / f"{stem}.keras"
    output_preprocess = MODELS / f"{stem}_preprocessing.npz"
    if candidate_path.exists() or output_preprocess.exists():
        raise FileExistsError("Candidate output already exists; retry to generate a new version")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(candidate_path), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7 if args.mode == "finetune" else 15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3 if args.mode == "finetune" else 6,
            factor=0.5, min_lr=1e-6
        ),
    ]
    print(f"\nMode: {args.mode}; LR: {learning_rate}; max epochs: {epochs}")
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weights, callbacks=callbacks, verbose=1,
    )

    best = tf.keras.models.load_model(candidate_path, compile=False)
    y_pred = np.argmax(best.predict(X_test, batch_size=256, verbose=0), axis=1)
    metrics = {
        "mode": args.mode,
        "base_model": args.base_model if args.mode == "finetune" else None,
        "model_file": candidate_path.name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "epochs_completed": len(history.history["loss"]),
        "note": "Kaggle random split is a development metric, not independent webcam performance. "
                "For a strict evaluation, use a saved split manifest and separate webcam sessions.",
    }
    np.savez(output_preprocess, mean=mean, std=std, classes=np.array(LETTERS, dtype=str))
    with (RESULTS / f"{stem}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (RESULTS / f"{stem}_report.txt").open("w", encoding="utf-8") as file:
        file.write(classification_report(y_test, y_pred, labels=range(24),
                                         target_names=LETTERS, zero_division=0))
    np.savetxt(
        RESULTS / f"{stem}_confusion_matrix.csv",
        confusion_matrix(y_test, y_pred, labels=range(24)), delimiter=",", fmt="%d",
    )
    print("\nEvaluation:", json.dumps(metrics, indent=2))
    print("Candidate model:", candidate_path)
    print("Candidate preprocessing:", output_preprocess)
    print("Existing models were NOT overwritten.")


if __name__ == "__main__":
    main()
