import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from inspect_dataset import parse_digit_label

def calculate_speaker_split(raw_dir="data/raw", test_size=0.2, seed=42):
    raw_path = Path(raw_dir)
    wav_files = sorted(list(raw_path.rglob("*.wav")) + list(raw_path.rglob("*.WAV")))
    
    file_paths = []
    labels = []
    speaker_ids = []

    for f in wav_files:
        rel_path = f.relative_to(raw_path)
        if len(rel_path.parts) < 2:
            continue
            
        speaker = rel_path.parts[0]
        label = parse_digit_label(f.name)
        if label is None and len(rel_path.parts) > 2:
            label = parse_digit_label(rel_path.parts[1])

        if label is not None and 0 <= label <= 9:
            file_paths.append(str(rel_path))
            labels.append(label)
            speaker_ids.append(speaker)

    X = np.array(file_paths)
    y = np.array(labels)
    groups = np.array(speaker_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_speakers = set(groups[train_idx])
    test_speakers = set(groups[test_idx])
    overlap = train_speakers.intersection(test_speakers)

    print("=" * 65)
    print("PROPOSED SPEAKER-INDEPENDENT TRAIN / TEST SPLIT SUMMARY")
    print("=" * 65)
    print(f"Total Valid Labeled WAV Clips: {len(X)}")
    print(f"Total Unique Speakers:          {len(set(groups))}")
    print(f"Train Set: {len(train_idx):6d} clips ({len(train_idx)/len(X)*100:.1f}%) across {len(train_speakers)} speakers")
    print(f"Test Set:  {len(test_idx):6d} clips ({len(test_idx)/len(X)*100:.1f}%) across {len(test_speakers)} speakers")
    print(f"Speaker Overlap (Must be 0):    {len(overlap)}")

    print("\n--- Train vs Test Class Distribution ---")
    train_counts = Counter(y[train_idx])
    test_counts = Counter(y[test_idx])
    print(f"{'Digit':<8}{'Train Count':<14}{'Train %':<12}{'Test Count':<14}{'Test %':<12}")
    print("-" * 60)
    for digit in range(10):
        tr_c = train_counts[digit]
        te_c = test_counts[digit]
        tr_p = tr_c / len(train_idx) * 100
        te_p = te_c / len(test_idx) * 100
        print(f"Digit {digit:<2} {tr_c:<14} {tr_p:<11.1f}% {te_c:<14} {te_p:<11.1f}%")
    print("=" * 65)

if __name__ == "__main__":
    calculate_speaker_split()
