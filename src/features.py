import os
import joblib
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_features_single(file_path, sample_rate=16000, duration_sec=1.0, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Loads an audio file, converts to mono 16kHz, pads/crops to fixed length,
    extracts 13 MFCCs + Delta + Delta-Delta, and pools across time (mean + std).
    Returns a 78-dimensional feature vector.
    """
    try:
        # Load audio with soundfile / librosa
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        # Return zeros vector if file cannot be read
        return np.zeros(n_mfcc * 6, dtype=np.float32)

    # Pad or crop audio to fixed length (e.g., 16000 samples)
    target_samples = int(sample_rate * duration_sec)
    if len(y) < target_samples:
        pad_width = target_samples - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    else:
        y = y[:target_samples]

    # Handle silent/empty clips safely
    if np.max(np.abs(y)) < 1e-6:
        return np.zeros(n_mfcc * 6, dtype=np.float32)

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Compute Delta and Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Summary statistics across time dimension (mean and std)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)

    # Concatenate into 78-dimensional feature vector
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std
    ]).astype(np.float32)

    return feature_vector

def _worker_extract(item, audio_config):
    full_path = item["full_path"]
    label = item["label"]
    feat = extract_features_single(
        full_path,
        sample_rate=audio_config["sample_rate"],
        duration_sec=audio_config["duration_sec"],
        n_mfcc=audio_config["n_mfcc"],
        n_fft=audio_config["n_fft"],
        hop_length=audio_config["hop_length"]
    )
    return feat, label

def extract_features_batch(items, audio_config, desc="Extracting Features"):
    features = []
    labels = []
    
    # Use max available CPU threads for fast extraction
    max_workers = min(os.cpu_count() or 4, 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker_extract, item, audio_config) for item in items]
        for future in tqdm(as_completed(futures), total=len(items), desc=desc):
            feat, label = future.result()
            features.append(feat)
            labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

def load_or_extract_features(manifest, config, force_reextract=False):
    cache_path = Path(config["dataset"]["feature_cache_path"])
    if cache_path.exists() and not force_reextract:
        print(f"Loading cached MFCC features from: {cache_path}")
        data = joblib.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    print(f"Extracting 78-dim MFCC+Delta+Delta2 features for dataset...")
    audio_config = config["audio"]

    X_train, y_train = extract_features_batch(manifest["train"], audio_config, desc="Extracting Train MFCCs")
    X_test, y_test = extract_features_batch(manifest["test"], audio_config, desc="Extracting Test MFCCs")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }, cache_path)

    print(f"Saved feature cache to: {cache_path}")
    return X_train, y_train, X_test, y_test
