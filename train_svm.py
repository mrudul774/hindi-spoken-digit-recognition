import os
import yaml
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.dataset import get_or_create_manifest
from src.features import load_or_extract_features

def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("HINDI SPOKEN DIGIT RECOGNITION - V1 SVM BASELINE TRAINING")
    print("=" * 65)

    # 1. Manifest creation / loading
    manifest = get_or_create_manifest(config)

    # 2. Feature Extraction / loading
    X_train, y_train, X_test, y_test = load_or_extract_features(manifest, config)

    print(f"\nDataset Feature Matrix Shapes:")
    print(f"  - X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  - X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    # 3. Feature Scaling
    print("\nFitting StandardScaler on training set features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Support Vector Machine Classifier
    kernel = config["model"].get("kernel", "rbf")
    C = config["model"].get("C", 10.0)
    print(f"Training Support Vector Machine (SVC kernel='{kernel}', C={C})...")
    
    clf = SVC(kernel=kernel, C=C, probability=True, random_state=config["dataset"]["random_seed"])
    clf.fit(X_train_scaled, y_train)

    train_acc = clf.score(X_train_scaled, y_train) * 100
    test_acc = clf.score(X_test_scaled, y_test) * 100

    print(f"\nTraining Completed Successfully:")
    print(f"  - Train Accuracy: {train_acc:.2f}%")
    print(f"  - Test Accuracy:  {test_acc:.2f}% (Speaker-Independent Evaluation)")

    # 5. Save model checkpoint
    model_path = Path(config["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        "scaler": scaler,
        "model": clf,
        "config": config,
        "train_acc": train_acc,
        "test_acc": test_acc
    }, model_path)
    
    print(f"Saved trained model checkpoint to: {model_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
