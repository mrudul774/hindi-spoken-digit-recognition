import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from src.dataset import get_or_create_manifest
from src.features_2d import load_or_extract_2d_tensors
from src.models.cnn2d import LightweightAudioCNN2D, count_parameters

DIGIT_NAMES = ["0 (shunya)", "1 (ek)", "2 (do)", "3 (teen)", "4 (chaar)",
               "5 (paanch)", "6 (chhah)", "7 (saat)", "8 (aath)", "9 (nau)"]

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path("models/cnn2d_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN model checkpoint not found at {checkpoint_path}. Please run train_cnn2d.py first.")

    meta_path = Path("models/cnn2d_meta.json")
    meta_info = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta_info = json.load(f)

    print("=" * 70)
    print("HINDI SPOKEN DIGIT RECOGNITION - V2 2D CNN EVALUATION")
    print("=" * 70)

    # 1. Load Checkpoint & Normalization Stats
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    mean_ch = checkpoint["mean_ch"]
    std_ch = checkpoint["std_ch"]

    # 2. Load Exact Same Test Manifest & 2D Tensors
    manifest = get_or_create_manifest(config)
    _, _, X_test_raw, y_test_raw = load_or_extract_2d_tensors(manifest, config)

    # 3. Normalize Test Data using Training Set Mean/Std
    X_test_norm = (X_test_raw - mean_ch) / (std_ch + 1e-6)

    test_ds = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test_raw, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 4. Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightAudioCNN2D(num_classes=10, in_channels=3, n_mfcc=40)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # 5. Predict on Test Set
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())

    y_test = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 6. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_test, y_pred, average=None)

    total_params = checkpoint.get("total_params", count_parameters(model))
    train_time_sec = meta_info.get("training_time_sec", 0.0)

    print(f"\n--- Model Complexity & Training Specs ---")
    print(f"  - Model Architecture:      Lightweight 2D ConvNet")
    print(f"  - Total Parameters:        {total_params:,} (~0.58M params)")
    print(f"  - Total Training Time:     {train_time_sec:.2f} seconds")

    print("\n--- OVERALL METRICS SUMMARY (SPEAKER-INDEPENDENT TEST SET) ---")
    print(f"  - Test Accuracy:          {acc * 100:.2f}%")
    print(f"  - Precision (Macro):      {p_macro * 100:.2f}%")
    print(f"  - Precision (Weighted):   {p_weighted * 100:.2f}%")
    print(f"  - Recall (Macro):         {r_macro * 100:.2f}%")
    print(f"  - Recall (Weighted):      {r_weighted * 100:.2f}%")
    print(f"  - F1 Score (Macro):       {f1_macro * 100:.2f}%")
    print(f"  - F1 Score (Weighted):    {f1_weighted * 100:.2f}%")

    print("\n--- DETAILED PER-CLASS CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=DIGIT_NAMES, digits=4))

    # 7. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_png_path = Path("results/confusion_matrix_cnn2d.png")
    cm_png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=[f"Digit {i}" for i in range(10)],
                yticklabels=[f"Digit {i}" for i in range(10)])
    plt.title("Hindi Spoken Digit Recognition - V2 2D CNN Confusion Matrix\n(Speaker-Independent Test Set)")
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Ground-Truth Digit")
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_png_path}")

    # 8. Save Metrics Artifact JSON
    results_json_path = Path("results/cnn2d_results.json")
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
            "model_name": "Lightweight 2D CNN",
            "total_parameters": int(total_params),
            "training_time_sec": float(train_time_sec),
            "test_speakers": manifest["metadata"]["test_speakers"],
            "test_samples": len(y_test)
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

    print(f"Saved CNN evaluation metrics JSON to: {results_json_path}")

    # 9. Load SVM Baseline Results for Direct Side-by-Side Comparison
    svm_results_path = Path("results/svm_baseline_results.json")
    if svm_results_path.exists():
        with open(svm_results_path, "r") as f:
            svm_res = json.load(f)["metrics"]
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY (IDENTICAL 22-SPEAKER TEST SET)")
        print("=" * 70)
        print(f"{'Model':<25}{'Accuracy':<14}{'Macro Prec':<14}{'Macro Rec':<14}{'Macro F1':<14}")
        print("-" * 70)
        print(f"{'V1 SVM Baseline':<25}{svm_res['accuracy']*100:<13.2f}%{svm_res['precision_macro']*100:<13.2f}%{svm_res['recall_macro']*100:<13.2f}%{svm_res['f1_macro']*100:<13.2f}%")
        print(f"{'V2 Lightweight 2D CNN':<25}{acc*100:<13.2f}%{p_macro*100:<13.2f}%{r_macro*100:<13.2f}%{f1_macro*100:<13.2f}%")
        diff_acc = (acc - svm_res['accuracy']) * 100
        print("-" * 70)
        print(f"Accuracy Gain (CNN vs SVM): {diff_acc:+.2f} percentage points")
        print("=" * 70)

if __name__ == "__main__":
    main()
