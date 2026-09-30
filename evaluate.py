import os
import json
import yaml
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from src.dataset import get_or_create_manifest
from src.features import load_or_extract_features

DIGIT_NAMES = ["0 (shunya)", "1 (ek)", "2 (do)", "3 (teen)", "4 (chaar)",
               "5 (paanch)", "6 (chhah)", "7 (saat)", "8 (aath)", "9 (nau)"]

def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = Path(config["output"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train_svm.py first.")

    print("=" * 70)
    print("HINDI SPOKEN DIGIT RECOGNITION - EVALUATION & METRICS REPORT")
    print("=" * 70)

    # 1. Load model checkpoint
    checkpoint = joblib.load(model_path)
    scaler = checkpoint["scaler"]
    clf = checkpoint["model"]

    # 2. Load manifest and test data
    manifest = get_or_create_manifest(config)
    X_train, y_train, X_test, y_test = load_or_extract_features(manifest, config)

    # 3. Predict on Test Set
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    # 4. Calculate Overall & Per-Class Metrics
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    # Detailed per-class precision, recall, f1, support
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_test, y_pred, average=None)

    print("\n--- OVERALL METRICS SUMMARY (SPEAKER-INDEPENDENT TEST SET) ---")
    print(f"  - Test Accuracy:          {acc * 100:.2f}%")
    print(f"  - Precision (Macro):      {p_macro * 100:.2f}%")
    print(f"  - Precision (Weighted):   {p_weighted * 100:.2f}%")
    print(f"  - Recall (Macro):         {r_macro * 100:.2f}%")
    print(f"  - Recall (Weighted):      {r_weighted * 100:.2f}%")
    print(f"  - F1 Score (Macro):       {f1_macro * 100:.2f}%")
    print(f"  - F1 Score (Weighted):    {f1_weighted * 100:.2f}%")

    print("\n--- DETAILED PER-CLASS CLASSIFICATION REPORT ---")
    report_str = classification_report(y_test, y_pred, target_names=DIGIT_NAMES, digits=4)
    print(report_str)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix Heatmap
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_png_path = Path(config["output"]["confusion_matrix_png"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Digit {i}" for i in range(10)],
                yticklabels=[f"Digit {i}" for i in range(10)])
    plt.title("Hindi Spoken Digit Recognition - SVM Baseline Confusion Matrix\n(Speaker-Independent Test Set)")
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Ground-Truth Digit")
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_png_path}")

    # 6. Save JSON Metrics Artifact
    results_json_path = Path(config["output"]["results_json"])
    per_class_dict = {}
    for i in range(10):
        per_class_dict[f"digit_{i}"] = {
            "name": DIGIT_NAMES[i],
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support_class[i])
        }

    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_type": config["model"]["type"],
            "kernel": config["model"]["kernel"],
            "C": config["model"]["C"],
            "test_speakers": manifest["metadata"]["test_speakers"],
            "train_speakers": manifest["metadata"]["train_speakers"],
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        },
        "metrics": {
            "accuracy": float(acc),
            "precision_macro": float(p_macro),
            "precision_weighted": float(p_weighted),
            "recall_macro": float(r_macro),
            "recall_weighted": float(r_weighted),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted)
        },
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }

    with open(results_json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Saved exact evaluation metrics JSON to: {results_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
