import os
import time
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from src.dataset import get_or_create_manifest
from src.features_2d import load_or_extract_2d_tensors
from src.models.cnn2d import LightweightAudioCNN2D, count_parameters

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("HINDI SPOKEN DIGIT RECOGNITION - V2 2D CNN TRAINING (PyTorch)")
    print("=" * 70)

    # 1. Load exact same speaker-independent manifest
    manifest = get_or_create_manifest(config)

    # 2. Load / Extract 2D Tensors (3, 40, T)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_or_extract_2d_tensors(manifest, config)

    print(f"\nRaw Tensor Shapes:")
    print(f"  - X_train: {X_train_raw.shape} | y_train: {y_train_raw.shape}")
    print(f"  - X_test:  {X_test_raw.shape}  | y_test:  {y_test_raw.shape}")

    # 3. Speaker-Independent Inner Train / Validation Split FIRST
    train_items = manifest["train"]
    train_speakers = np.array([item["speaker"] for item in train_items])

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    inner_train_idx, val_idx = next(gss_val.split(X_train_raw, y_train_raw, train_speakers))

    X_tr_raw, y_tr = X_train_raw[inner_train_idx], y_train_raw[inner_train_idx]
    X_val_raw, y_val = X_train_raw[val_idx], y_train_raw[val_idx]

    val_spk_count = len(set(train_speakers[val_idx]))
    tr_spk_count = len(set(train_speakers[inner_train_idx]))
    print(f"\nSpeaker-Independent Inner Train/Val Split:")
    print(f"  - Inner Train: {len(X_tr_raw):6d} clips across {tr_spk_count} speakers")
    print(f"  - Validation:  {len(X_val_raw):6d} clips across {val_spk_count} speakers")
    print(f"  - Test Set:    {len(X_test_raw):6d} clips across {manifest['metadata']['test_speakers']} speakers")

    # 4. Compute Channel-Wise Normalization STRICTLY FROM INNER TRAINING DATA
    print("\nComputing Channel Normalization Statistics ONLY from Inner Training Data (X_tr)...")
    mean_ch = np.mean(X_tr_raw, axis=(0, 2, 3), keepdims=True)
    std_ch = np.std(X_tr_raw, axis=(0, 2, 3), keepdims=True)

    print(f"  - Channel Means: {mean_ch.squeeze()}")
    print(f"  - Channel Stds:  {std_ch.squeeze()}")

    X_tr = (X_tr_raw - mean_ch) / (std_ch + 1e-6)
    X_val = (X_val_raw - mean_ch) / (std_ch + 1e-6)
    X_test_norm = (X_test_raw - mean_ch) / (std_ch + 1e-6)

    # 5. Create PyTorch DataLoaders
    batch_size = 64
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test_raw, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 6. Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = LightweightAudioCNN2D(num_classes=10, in_channels=3, n_mfcc=40).to(device)
    total_params = count_parameters(model)
    print(f"Initialized 2D CNN Architecture (Trainable Params: {total_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 7. Training Loop with Early Stopping
    max_epochs = 30
    patience = 8
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_path = Path("models/cnn2d_best.pth")
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("\nStarting CNN Training Loop...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_b.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_b).sum().item()
            total_train += y_b.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item() * X_b.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == y_b).sum().item()
                total_val += y_b.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "mean_ch": mean_ch,
                "std_ch": std_ch,
                "best_val_acc": epoch_val_acc,
                "best_val_loss": epoch_val_loss,
                "total_params": total_params,
                "epoch": epoch
            }, best_model_path)
            print(f"  [Checkpoint] Saved new best model to {best_model_path} (Val Acc: {epoch_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (No validation loss improvement for {patience} consecutive epochs).")
                break

    training_time_sec = time.time() - start_time
    print(f"\nTraining Completed in {training_time_sec:.2f} seconds.")

    # 8. Save CNN Meta info
    meta_path = Path("models/cnn2d_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "training_time_sec": training_time_sec,
            "total_params": total_params,
            "best_val_loss": best_val_loss
        }, f, indent=2)

    print("=" * 70)

if __name__ == "__main__":
    main()
