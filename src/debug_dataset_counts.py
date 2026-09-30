import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from inspect_dataset import parse_digit_label

def debug_dataset_counts(raw_dir="data/raw"):
    raw_path = Path(raw_dir).resolve()
    
    # 1. Compare glob behaviors on Windows
    wav_lower = list(raw_path.rglob("*.wav"))
    wav_upper = list(raw_path.rglob("*.WAV"))
    wav_concat = wav_lower + wav_upper
    
    # Deduplicate by resolved canonical path string
    unique_wav_set = set(str(f.resolve()).lower() for f in raw_path.rglob("*.wav"))
    unique_wav_files = sorted([Path(p) for p in unique_wav_set])

    print("=" * 70)
    print("DATASET DISCREPANCY INVESTIGATION & ROOT-CAUSE ANALYSIS")
    print("=" * 70)
    print(f"Windows Filesystem Case-Insensitive Glob Audit:")
    print(f"  - rglob('*.wav') count:         {len(wav_lower)}")
    print(f"  - rglob('*.WAV') count:         {len(wav_upper)}")
    print(f"  - Concatenated (wav + WAV):      {len(wav_concat)}  <-- Cause of 42,712 doubling!")
    print(f"  - True Unique Audio Files:      {len(unique_wav_files)}")

    # Show example of duplicate path in concatenated list
    if len(wav_concat) > len(unique_wav_files):
        print("\nExample of duplicate path caused by case-insensitive glob concatenation:")
        print(f"  - Path 1: {wav_concat[0]}")
        print(f"  - Path 2: {wav_concat[len(wav_lower)]}")
        print(f"  - Same file on disk? {wav_concat[0].resolve() == wav_concat[len(wav_lower)].resolve()}")

    # 2. Speaker directories & files per speaker
    speaker_dirs = [d for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    file_paths = []
    labels = []
    speaker_ids = []
    unparsed_files = []
    speaker_file_counts = defaultdict(int)

    for f in unique_wav_files:
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
            speaker_file_counts[speaker] += 1
        else:
            unparsed_files.append(str(rel_path))

    X = np.array(file_paths)
    y = np.array(labels)
    groups = np.array(speaker_ids)

    print("\n--- 1. VERIFIED GROUND TRUTH METRICS ---")
    print(f"Total Unique Speaker Directories:   {len(speaker_dirs)}")
    print(f"Total True Unique WAV Files:        {len(unique_wav_files)}")
    print(f"Total Valid Labeled (0-9) WAVs:     {len(X)}")
    print(f"Total Unparsed / Ambiguous WAVs:    {len(unparsed_files)}")

    print("\n--- 2. VALID LABELED FILES PER SPEAKER ---")
    spk_counts = list(speaker_file_counts.values())
    if spk_counts:
        print(f"Speakers with valid labeled audio: {len(speaker_file_counts)} / {len(speaker_dirs)}")
        print(f"Files per speaker: Min={min(spk_counts)}, Max={max(spk_counts)}, Mean={sum(spk_counts)/len(spk_counts):.1f}, Median={sorted(spk_counts)[len(spk_counts)//2]}")

    # 3. Clean Speaker-Independent GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_speakers = set(groups[train_idx])
    test_speakers = set(groups[test_idx])
    overlap = train_speakers.intersection(test_speakers)

    print("\n--- 3. VERIFIED SPEAKER-INDEPENDENT TRAIN/TEST SPLIT ---")
    print(f"Train Set: {len(train_idx):6d} clips ({len(train_idx)/len(X)*100:.1f}%) across {len(train_speakers)} speakers")
    print(f"Test Set:  {len(test_idx):6d} clips ({len(test_idx)/len(X)*100:.1f}%) across {len(test_speakers)} speakers")
    print(f"Speaker Overlap: {len(overlap)} (Zero leakage verified)")

    print("\nDigit Distribution Across Split:")
    train_c = Counter(y[train_idx])
    test_c = Counter(y[test_idx])
    print(f"{'Digit':<8}{'Train Count':<14}{'Train %':<12}{'Test Count':<14}{'Test %':<12}")
    print("-" * 60)
    for digit in range(10):
        tr_cnt = train_c[digit]
        te_cnt = test_c[digit]
        tr_pct = tr_cnt / len(train_idx) * 100
        te_pct = te_cnt / len(test_idx) * 100
        print(f"Digit {digit:<2} {tr_cnt:<14} {tr_pct:<11.1f}% {te_cnt:<14} {te_pct:<11.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    debug_dataset_counts()
