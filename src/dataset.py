import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from src.inspect_dataset import parse_digit_label

def build_manifest(raw_dir="data/raw", test_size=0.2, random_seed=42):
    raw_path = Path(raw_dir).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found at '{raw_path}'")

    # Deduplicate files by canonical path string to prevent Windows case-insensitive glob doubling
    unique_wav_set = set(str(f.resolve()) for f in raw_path.rglob("*.wav"))
    unique_wav_files = sorted([Path(p) for p in unique_wav_set])

    records = []
    for f in unique_wav_files:
        try:
            rel_path = f.relative_to(raw_path)
        except ValueError:
            continue

        if len(rel_path.parts) < 2:
            continue

        speaker = rel_path.parts[0]
        label = parse_digit_label(f.name)
        if label is None and len(rel_path.parts) > 2:
            label = parse_digit_label(rel_path.parts[1])

        if label is not None and 0 <= label <= 9:
            records.append({
                "rel_path": str(rel_path).replace("\\", "/"),
                "full_path": str(Path(raw_dir) / rel_path).replace("\\", "/"),
                "label": int(label),
                "speaker": str(speaker)
            })

    if not records:
        raise ValueError(f"No valid labeled WAV files found in '{raw_path}'")

    paths = np.array([r["rel_path"] for r in records])
    labels = np.array([r["label"] for r in records])
    speakers = np.array([r["speaker"] for r in records])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(gss.split(paths, labels, speakers))

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    manifest = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "random_seed": random_seed,
            "test_size": test_size,
            "total_samples": len(records),
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "train_speakers": len(set(speakers[train_idx])),
            "test_speakers": len(set(speakers[test_idx]))
        },
        "train": train_records,
        "test": test_records
    }
    return manifest

def get_or_create_manifest(config):
    manifest_path = Path(config["dataset"]["manifest_path"])
    if manifest_path.exists():
        print(f"Loading existing split manifest from: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        print(f"Generating new speaker-independent manifest...")
        manifest = build_manifest(
            raw_dir=config["dataset"]["raw_dir"],
            test_size=config["dataset"]["test_size"],
            random_seed=config["dataset"]["random_seed"]
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved reproducible split manifest to: {manifest_path}")
    return manifest

if __name__ == "__main__":
    test_config = {
        "dataset": {
            "raw_dir": "data/raw",
            "manifest_path": "data/split_manifest.json",
            "test_size": 0.2,
            "random_seed": 42
        }
    }
    manifest = get_or_create_manifest(test_config)
    print("Manifest creation test successful.")
