"""Evaluate saved static MLP and dynamic LSTM without webcam or manual trial CSV.
Run: python src/evaluate_model.py
Results: results/final_result/{static_24,dynamic_jz_other}/
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'dataset'
MODELS = ROOT / 'models'
OUT = ROOT / 'results' / 'final_result'
STATIC_MODEL = MODELS / 'asl_24_finetuned_candidate.keras'
DYNAMIC_MODEL = MODELS / 'asl_jz_other_lstm.keras'
FEATURES = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]


def save_results(name, truth, predicted, labels, note, model_path):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(labels)))
    values = {
        'model': str(model_path.name), 'samples': int(len(truth)),
        'accuracy': float(accuracy_score(truth, predicted)),
        'macro_precision': float(precision_score(truth, predicted, labels=indices, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(truth, predicted, labels=indices, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(truth, predicted, labels=indices, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(truth, predicted, labels=indices, average='weighted', zero_division=0)),
        'evaluation_note': note,
    }
    report = classification_report(truth, predicted, labels=indices,
                                   target_names=labels, zero_division=0)
    detailed = classification_report(truth, predicted, labels=indices,
                                     target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=indices)
    with (folder / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(values, f, indent=2)
    (folder / 'classification_report.txt').write_text(report, encoding='utf-8')
    pd.DataFrame(detailed).T.to_csv(folder / 'classification_report.csv')
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(folder / 'confusion_matrix.csv')
    for normalized in (False, True):
        cm = matrix.astype(float)
        if normalized:
            totals = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, totals, out=np.zeros_like(cm), where=totals != 0)
        fig, ax = plt.subplots(figsize=(15, 13) if len(labels) > 3 else (7, 6))
        if normalized:
            display_matrix = np.asarray(cm, dtype=np.float64)
            number_format = '.2f'
        else:
            display_matrix = np.asarray(cm).astype(np.int64)
            number_format = 'd'

        ConfusionMatrixDisplay(
            confusion_matrix=display_matrix,
            display_labels=labels
        ).plot(
            ax=ax, cmap='Blues', values_format=number_format,
            xticks_rotation=90, colorbar=True, include_values=True,text_kw ={"fontsize" :6}
        )
        ax.set_title(f'{name}: ' + ('Normalized ' if normalized else '') + 'Confusion Matrix')
        fig.tight_layout()
        fig.savefig(folder / ('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'), dpi=170)
        plt.close(fig)
    print(f'\n{name}:\n{report}\nMetrics: {json.dumps(values, indent=2)}')
    print('Saved:', folder)


def evaluate_static():
    prep = STATIC_MODEL.with_name(STATIC_MODEL.stem + '_preprocessing.npz')
    with np.load(prep, allow_pickle=False) as z:
        mean, std = z['mean'].astype(np.float32), z['std'].astype(np.float32)
        labels = z['classes'].astype(str).tolist()
    df = pd.read_csv(DATA / 'asl_landmarks_full.csv')
    df = df[df['label'].isin(labels)].copy()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    # Reconstruct earlier 70/15/15 stratified Kaggle split. Only valid if
    # original training used identical ordering, filtering, and random seed.
    _, temp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
    _, test = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=42)
    x = test[FEATURES].to_numpy(dtype=np.float32)
    x = (x - mean) / np.where(std < 1e-6, 1.0, std)
    truth = test['label'].map({v:i for i,v in enumerate(labels)}).to_numpy(dtype=int)
    model = tf.keras.models.load_model(STATIC_MODEL, compile=False)
    if model.output_shape[-1] != len(labels):
        raise ValueError('Static class count does not match model output')
    predicted = np.argmax(model.predict(x, batch_size=256, verbose=0), axis=1)
    save_results('static_24', truth, predicted, labels,
                 'Reconstructed Kaggle random split; not independent webcam/hybrid evaluation. Split must match training exactly.', STATIC_MODEL)


def evaluate_dynamic():
    prep = DYNAMIC_MODEL.with_name(DYNAMIC_MODEL.stem.replace('_lstm', '') + '_preprocessing.npz')
    with np.load(prep, allow_pickle=False) as z:
        mean, std = z['mean'].astype(np.float32), z['std'].astype(np.float32)
        labels = z['classes'].astype(str).tolist()
        version = str(z['feature_version'].item())
    if version != 'wrist_xy_delta_v1':
        raise ValueError(f'Unsupported dynamic feature version: {version}')
    data, targets = [], []
    for label in labels:
        paths = sorted((DATA / 'dynamic' / 'val' / label).rglob('*.npy'))
        for path in paths:
            seq = np.load(path, allow_pickle=False).astype(np.float32)
            if seq.shape != (30, 66) or not np.isfinite(seq).all():
                raise ValueError(f'Invalid validation sequence: {path}')
            data.append(seq)
            targets.append(labels.index(label))
    if not data or len(set(targets)) != len(labels):
        raise ValueError('Validation recordings for J, Z and OTHER are required')
    x = np.stack(data)
    x[:, :, 63:65] -= x[:, :1, 63:65]
    x[:, :, 65] = 0.0
    x = (x - mean) / np.where(std < 1e-6, 1.0, std)
    model = tf.keras.models.load_model(DYNAMIC_MODEL, compile=False)
    if model.output_shape[-1] != len(labels):
        raise ValueError('Dynamic class count does not match model output')
    predicted = np.argmax(model.predict(x, batch_size=32, verbose=0), axis=1)
    save_results('dynamic_jz_other', np.array(targets), predicted, labels,
                 'Webcam validation data used for model selection; not an untouched final test or hybrid A-Z evaluation.', DYNAMIC_MODEL)


if __name__ == '__main__':
    evaluate_static()
    evaluate_dynamic()
    print('\nDone. All outputs are inside:', OUT)
    print('A true combined A-Z matrix requires saved final hybrid-controller predictions.')
