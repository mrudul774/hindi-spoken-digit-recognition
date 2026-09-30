import os
import joblib
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_2d_tensor_single(file_path, sample_rate=16000, duration_sec=1.0, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Loads audio, resamples to 16kHz mono, pads/crops to fixed 1.0s length,
    extracts 40 MFCCs, Delta, and Delta-Delta.
    Returns 3D Tensor array of shape (3, 40, T) where T=32 frames.
    """
    target_samples = int(sample_rate * duration_sec)
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception:
        return np.zeros((3, n_mfcc, target_samples // hop_length + 1), dtype=np.float32)

    if len(y) < target_samples:
        pad_width = target_samples - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    else:
        y = y[:target_samples]

    if np.max(np.abs(y)) < 1e-6:
        T_expected = target_samples // hop_length + 1
        return np.zeros((3, n_mfcc, T_expected), dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    T_target = target_samples // hop_length + 1
    if mfcc.shape[1] < T_target:
        mfcc = np.pad(mfcc, ((0, 0), (0, T_target - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :T_target]

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    tensor = np.stack([mfcc, delta, delta2], axis=0).astype(np.float32)
    return tensor


def _worker_2d(args):
    """
    Worker that receives (index, item, audio_config) and returns (index, tensor, label).
    Returning the original index lets us reconstruct deterministic ordering after
    as_completed() delivers futures in non-deterministic completion order.
    """
    idx, item, audio_config = args
    full_path = item["full_path"]
    label = item["label"]
    tensor = extract_2d_tensor_single(
        full_path,
        sample_rate=audio_config.get("sample_rate", 16000),
        duration_sec=audio_config.get("duration_sec", 1.0),
        n_mfcc=audio_config.get("n_mfcc_2d", 40),
        n_fft=audio_config.get("n_fft", 2048),
        hop_length=audio_config.get("hop_length", 512)
    )
    return idx, tensor, label


def extract_2d_tensors_batch(items, audio_config, desc="Extracting 2D Spectral Tensors"):
    """
    Extracts features in DETERMINISTIC ORDER matching the manifest item list.

    Design:
    - Each worker receives its original input index alongside the item.
    - Each worker returns (original_index, tensor, label).
    - Results are placed back at tensors[original_index] and labels[original_index].
    - This guarantees output[i] always corresponds exactly to items[i],
      regardless of the non-deterministic completion order of as_completed().
    """
    n = len(items)
    tensors = [None] * n
    labels = [None] * n

    max_workers = min(os.cpu_count() or 4, 8)

    # Package each task as (original_index, item, config)
    tasks = [(i, item, audio_config) for i, item in enumerate(items)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_2d, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=n, desc=desc):
            idx, tensor, label = future.result()
            tensors[idx] = tensor
            labels[idx] = label

    return np.array(tensors, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_or_extract_2d_tensors(manifest, config, force_reextract=False):
    cache_path = Path("data/processed/mfcc_2d_tensors.joblib")
    if cache_path.exists() and not force_reextract:
        print(f"Loading cached 2D spectral tensors from: {cache_path}")
        data = joblib.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    print(f"Extracting 2D (3, 40, T) spectral tensors for CNN...")
    audio_config = config["audio"].copy()   # copy to avoid mutating original config dict
    audio_config["n_mfcc_2d"] = 40         # 40 MFCC bins for 2D ConvNet

    X_train, y_train = extract_2d_tensors_batch(manifest["train"], audio_config, desc="Extracting Train 2D Tensors")
    X_test, y_test = extract_2d_tensors_batch(manifest["test"], audio_config, desc="Extracting Test 2D Tensors")

    # Verify label alignment with manifest before saving
    for split_name, items, labels in [("train", manifest["train"], y_train), ("test", manifest["test"], y_test)]:
        for i, (item, label) in enumerate(zip(items, labels)):
            if item["label"] != int(label):
                raise RuntimeError(
                    f"Label alignment error in {split_name} at index {i}: "
                    f"manifest says {item['label']}, feature array has {int(label)}"
                )
    print("  [OK] Label alignment verified: all features match manifest order.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }, cache_path)

    print(f"Saved 2D tensor cache to: {cache_path}")
    return X_train, y_train, X_test, y_test
