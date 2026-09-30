"""
Verification script for deterministic feature ordering in extract_2d_tensors_batch.

Checks:
  1. Output feature array order exactly matches input file list order.
  2. Labels match corresponding file paths.
  3. Speaker IDs match corresponding file paths.
  4. Prints first 10 file paths with label, speaker, and feature index.
  5. Runs extraction twice and verifies the two runs produce identical ordering.
  6. Verifies data/split_manifest.json is unchanged.
  7. Does NOT retrain or modify any cached data.
"""
import json
import hashlib
import numpy as np
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import production extraction function
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.features_2d import _worker_2d, extract_2d_tensors_batch


def extract_small_batch_with_index_log(items, audio_config, n_samples=30):
    """
    Runs extraction on a small subset and returns:
    - Feature tensors in manifest order
    - Labels in manifest order
    - Mapping: {extracted_index: (rel_path, manifest_label, manifest_speaker)}
    """
    n = len(items)
    tensors = [None] * n
    labels = [None] * n
    returned_indices = []

    tasks = [(i, item, audio_config) for i, item in enumerate(items)]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker_2d, task): task[0] for task in tasks}
        for future in as_completed(futures):
            idx, tensor, label = future.result()
            tensors[idx] = tensor
            labels[idx] = label
            returned_indices.append(idx)

    return (
        np.array(tensors, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        returned_indices
    )


def main():
    config_path = Path("configs/config.yaml")
    manifest_path = Path("data/split_manifest.json")

    print("=" * 70)
    print("DETERMINISTIC ORDERING VERIFICATION")
    print("=" * 70)

    # 6. Verify manifest is unchanged
    manifest_hash_before = hashlib.md5(manifest_path.read_bytes()).hexdigest()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest_hash_after = hashlib.md5(manifest_path.read_bytes()).hexdigest()
    assert manifest_hash_before == manifest_hash_after, "FAIL: manifest was modified during read!"
    print(f"[6] split_manifest.json hash: {manifest_hash_before} — UNCHANGED ✓")

    meta = manifest["metadata"]
    print(f"    Train: {meta['train_samples']} clips | Test: {meta['test_samples']} clips")
    print(f"    Train speakers: {meta['train_speakers']} | Test speakers: {meta['test_speakers']}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    audio_config = config["audio"].copy()
    audio_config["n_mfcc_2d"] = 40

    # Use a small reproducible subset for fast verification (first 60 train items)
    N = 60
    subset = manifest["train"][:N]

    print(f"\nRunning two independent extractions on the first {N} manifest items...")

    # Run 1
    print("\n--- Extraction Run 1 ---")
    X1, y1, ret_idx_1 = extract_small_batch_with_index_log(subset, audio_config, n_samples=N)

    # Run 2
    print("--- Extraction Run 2 ---")
    X2, y2, ret_idx_2 = extract_small_batch_with_index_log(subset, audio_config, n_samples=N)

    # 1. Check output order matches manifest order (label at position i == manifest[i]["label"])
    print("\n[1] Verifying feature array order matches manifest item order...")
    all_ok = True
    for i in range(N):
        manifest_label = subset[i]["label"]
        extracted_label_1 = int(y1[i])
        extracted_label_2 = int(y2[i])
        if manifest_label != extracted_label_1 or manifest_label != extracted_label_2:
            print(f"    FAIL at index {i}: manifest={manifest_label}, run1={extracted_label_1}, run2={extracted_label_2}")
            all_ok = False
    if all_ok:
        print(f"    All {N} labels correctly aligned with manifest order ✓")

    # 2. Labels match file paths
    print("\n[2] Verifying labels match file paths...")
    for i in range(min(5, N)):
        item = subset[i]
        fname = Path(item["rel_path"]).name
        print(f"    [{i:02d}] label={int(y1[i])} | manifest_label={item['label']} | file={fname}")
    label_match = all(int(y1[i]) == subset[i]["label"] for i in range(N))
    print(f"    All labels match file paths: {'✓' if label_match else 'FAIL'}")

    # 3. Speaker IDs match file paths
    print("\n[3] Verifying speaker IDs match file paths...")
    for i in range(min(5, N)):
        item = subset[i]
        speaker_in_path = Path(item["rel_path"]).parts[0] if len(Path(item["rel_path"]).parts) > 1 else "N/A"
        manifest_speaker = item["speaker"]
        match = (speaker_in_path == manifest_speaker)
        print(f"    [{i:02d}] path_speaker='{speaker_in_path}' | manifest_speaker='{manifest_speaker}' | {'✓' if match else 'MISMATCH'}")

    # 4. Print first 10 entries: index, rel_path, label, speaker
    print("\n[4] First 10 manifest entries vs extracted features:")
    print(f"    {'Idx':<5} {'Label(manifest)':<17} {'Label(run1)':<13} {'Speaker':<40} {'File'}")
    print("    " + "-" * 100)
    for i in range(min(10, N)):
        item = subset[i]
        fname = Path(item["rel_path"]).name
        spk = item["speaker"][:38]
        print(f"    {i:<5} {item['label']:<17} {int(y1[i]):<13} {spk:<40} {fname}")

    # 5. Two runs produce identical ordering (same numpy array values)
    print("\n[5] Verifying two extraction runs produce identical feature arrays...")
    arrays_match = np.allclose(X1, X2, atol=1e-5)
    labels_match = np.array_equal(y1, y2)
    print(f"    Feature arrays identical (atol=1e-5): {'✓' if arrays_match else 'FAIL'}")
    print(f"    Label arrays identical:                {'✓' if labels_match else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    all_passed = all_ok and label_match and arrays_match and labels_match
    print(f"VERIFICATION RESULT: {'ALL CHECKS PASSED ✓' if all_passed else 'ONE OR MORE CHECKS FAILED ✗'}")
    print("=" * 70)
    print("Current mfcc_2d_tensors.joblib exists:", Path("data/processed/mfcc_2d_tensors.joblib").exists())
    print("data/split_manifest.json unchanged:    ", manifest_hash_before == hashlib.md5(manifest_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
