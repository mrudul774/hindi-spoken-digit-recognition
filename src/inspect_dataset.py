import os
import sys
import wave
import glob
import re
from pathlib import Path
from collections import defaultdict, Counter

# Hindi digit mapping dictionary for text/romanized filenames
HINDI_DIGIT_MAP = {
    # Numerical
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    # English names
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    # Devanagari / Romanized Hindi digit spellings
    'shunya': 0, 'shunyaa': 0, 'sunya': 0, 'zero': 0,
    'ek': 1, 'aek': 1, 'one': 1,
    'do': 2, 'doo': 2, 'two': 2,
    'teen': 3, 'tin': 3, 'three': 3,
    'chaar': 4, 'char': 4, 'four': 4,
    'paanch': 5, 'panch': 5, 'paanach': 5, 'five': 5,
    'chhah': 6, 'chah': 6, 'che': 6, 'chh': 6, 'cheh': 6, 'six': 6,
    'saat': 7, 'sat': 7, 'seven': 7,
    'aath': 8, 'ath': 8, 'aat': 8, 'eight': 8,
    'nau': 9, 'nao': 9, 'no': 9, 'nine': 9
}

def parse_digit_label(filename_or_folder):
    """
    Extract digit label 0-9 from the observed dataset filename conventions.

    Unknown or ambiguous conventions return None.
    """

    stem = Path(filename_or_folder).stem.lower().strip()

    # ---------------------------------------------------------
    # 1. Label-only filename: 0.wav ... 9.wav
    # ---------------------------------------------------------
    if re.fullmatch(r"[0-9]", stem):
        return int(stem)

    # ---------------------------------------------------------
    # 2. Standard:
    #    <speaker>_<digit>_<repetition>
    #
    #    speaker_2_01.wav -> 2
    # ---------------------------------------------------------
    match = re.search(r"_(\d)_(?:\d+)$", stem)
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 3. Zero-padded digit:
    #    <speaker>_<00-09>_<repetition>
    #
    #    speaker_08_01.wav -> 8
    # ---------------------------------------------------------
    match = re.search(r"_(0\d)_(?:\d+)$", stem)
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 4. Digit + repetition with duplicate suffix:
    #
    #    speaker_0_03 (1).wav -> 0
    # ---------------------------------------------------------
    match = re.search(r"_(\d)\s*\(\d+\)$", stem)
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 5. Hyphenated:
    #
    #    speaker_6-17.wav -> 6
    # ---------------------------------------------------------
    match = re.search(r"_(\d)-\d+$", stem)
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 6. Written digit:
    #
    #    speaker_eight (1).wav -> 8
    # ---------------------------------------------------------
    match = re.search(r"_([a-z]+)\s*\(\d+\)$", stem)
    if match:
        token = match.group(1)
        if token in HINDI_DIGIT_MAP:
            return HINDI_DIGIT_MAP[token]

    # ---------------------------------------------------------
    # 7. Written digit with hyphen:
    #
    #    speaker_zero- (1).wav -> 0
    # ---------------------------------------------------------
    match = re.search(r"_([a-z]+)\s*-\s*\(\d+\)$", stem)
    if match:
        token = match.group(1)
        if token in HINDI_DIGIT_MAP:
            return HINDI_DIGIT_MAP[token]

    # ---------------------------------------------------------
    # 8. Written digit + repetition + date:
    #
    #    speaker_eight10_09-10-22_...
    # ---------------------------------------------------------
    match = re.search(
        r"_([a-z]+)\d+_\d{2}-\d{2}-\d{2}_",
        stem
    )
    if match:
        token = match.group(1)
        if token in HINDI_DIGIT_MAP:
            return HINDI_DIGIT_MAP[token]

    # ---------------------------------------------------------
    # 9. Written digit with hyphen + repetition:
    #
    #    speaker_zero-02.wav -> 0
    # ---------------------------------------------------------
    match = re.search(r"_([a-z]+)-\d+$", stem)
    if match:
        token = match.group(1)
        if token in HINDI_DIGIT_MAP:
            return HINDI_DIGIT_MAP[token]

    # ---------------------------------------------------------
    # 10. Telugu/Hindi transliterated aliases observed:
    #
    #    shoony_01.wav -> 0
    #    paach_07.wav  -> 5
    # ---------------------------------------------------------
    alias_map = {
        "shoony": 0,
        "paach": 5,
    }

    match = re.search(r"_([a-z]+)_\d+$", stem)
    if match:
        token = match.group(1)
        if token in alias_map:
            return alias_map[token]

    # ---------------------------------------------------------
    # 11. Date/time:
    #
    #    speaker_8_09-10-22_...
    # ---------------------------------------------------------
    match = re.search(
        r"_(\d)_\d{2}-\d{2}-\d{2}_",
        stem
    )
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 12. Digit + repetition encoded together before date:
    #
    #    speaker_01_08-10-22_... -> digit 0
    #    speaker_57_08-10-22_... -> digit 5
    # ---------------------------------------------------------
    match = re.search(
        r"_(\d)(\d)_\d{2}-\d{2}-\d{2}_",
        stem
    )
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 13. Three-digit digit+repetition before date:
    #
    #    speaker_010_08-10-22_... -> 0
    #    speaker_110_08-10-22_... -> 1
    #    ...
    #    speaker_910_08-10-22_... -> 9
    # ---------------------------------------------------------
    match = re.search(
        r"_(\d)\d{2}_\d{2}-\d{2}-\d{2}_",
        stem
    )
    if match:
        return int(match.group(1))

    # ---------------------------------------------------------
    # 14. Digit + repetition + literal "wav" + date:
    #
    #    speaker_08_01_wav_11-09-22_...
    #    speaker_0_01wav_07-09-22_...
    #
    # Accept only a single digit or zero-padded 0-9 label.
    # This deliberately rejects "_16_wav_..." because 16
    # is not a valid class.
    # ---------------------------------------------------------
    match = re.search(
        r"_(0?[0-9])_(\d+)_?wav_",
        stem
    )
    if match:
        label_token = match.group(1)
        return int(label_token)

    # ---------------------------------------------------------
    # Unknown / ambiguous naming convention
    # ---------------------------------------------------------
    return None
def inspect_raw_dataset(raw_dir):
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print(f"Error: Directory '{raw_dir}' does not exist.")
        return False

    # 1. Speaker Folders
    speaker_dirs = [d for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # Find all audio files (check wav, m4a, mp3, flac, ogg)
    audio_extensions = ['*.wav', '*.WAV', '*.mp3', '*.MP3', '*.m4a', '*.M4A', '*.ogg', '*.OGG', '*.flac']
    all_audio_files = []
    for ext in audio_extensions:
        all_audio_files.extend(list(raw_path.rglob(ext)))

    all_audio_files = sorted(list(set(all_audio_files)))
    wav_files = [f for f in all_audio_files if f.suffix.lower() == '.wav']
    non_wav_audio = [f for f in all_audio_files if f.suffix.lower() != '.wav']
    zip_files = list(raw_path.rglob('*.zip')) + list(raw_path.rglob('*.ZIP'))

    print("=" * 70)
    print("HINDI SPOKEN DIGIT RECOGNITION - DETAILED DATASET INSPECTION")
    print("=" * 70)
    print(f"Dataset root location: {raw_path.resolve()}")
    print(f"Total Speaker Subdirectories: {len(speaker_dirs)}")
    print(f"Total WAV Audio Files Found:  {len(wav_files)}")
    if non_wav_audio:
        print(f"Other Audio Formats Found:    {len(non_wav_audio)} ({set(f.suffix for f in non_wav_audio)})")
    if zip_files:
        print(f"Compressed Archives (.zip):   {len(zip_files)} (Warning: zip files present inside raw dataset)")

    # 2. Extract digit labels and speaker associations
    digit_counts = Counter()
    unparsed_files = []
    speaker_file_map = defaultdict(list)
    sample_encoding_examples = defaultdict(list)

    for f in wav_files:
        rel_path = f.relative_to(raw_path)
        speaker_id = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        speaker_file_map[speaker_id].append(f)

        # Try parsing digit label from file stem first, then parent folders
        label = parse_digit_label(f.name)
        if label is None and len(rel_path.parts) > 2:
            # Check inner folder name
            label = parse_digit_label(rel_path.parts[1])

        if label is not None and 0 <= label <= 9:
            digit_counts[label] += 1
            if len(sample_encoding_examples[label]) < 3:
                sample_encoding_examples[label].append(str(rel_path))
        else:
            unparsed_files.append(rel_path)

    print("\n--- 1. DIGIT LABEL ENCODING & RECORDINGS PER DIGIT ---")
    print(f"Successfully labeled WAV files: {sum(digit_counts.values())} / {len(wav_files)}")
    if unparsed_files:
        print(f"Unparsed / Ambiguous WAV files: {len(unparsed_files)}")
        print("Sample unparsed filenames:")
        for u in unparsed_files[:10]:
            print(f"  - {u}")

    print("\nRecordings Per Digit Class (0-9):")
    for digit in range(10):
        count = digit_counts[digit]
        pct = (count / len(wav_files) * 100) if wav_files else 0
        print(f"  Digit '{digit}': {count:4d} files ({pct:5.1f}%)")

    print("\nSample Encoding Patterns per Digit:")
    for digit in range(10):
        examples = sample_encoding_examples[digit]
        ex_str = " | ".join(examples) if examples else "None found"
        print(f"  Digit '{digit}': {ex_str}")

    # 3. Speaker Reliability Analysis
    print("\n--- 2. SPEAKER ID EXTRACTION RELIABILITY ---")
    print(f"Total Unique Speaker Directories: {len(speaker_dirs)}")
    speakers_with_files = {spk: files for spk, files in speaker_file_map.items() if spk != "root"}
    print(f"Speaker directories containing WAV audio: {len(speakers_with_files)}")
    
    files_per_speaker = [len(files) for files in speakers_with_files.values()]
    if files_per_speaker:
        print(f"Audio files per speaker directory: min={min(files_per_speaker)}, max={max(files_per_speaker)}, mean={sum(files_per_speaker)/len(files_per_speaker):.1f}")
    
    print("\nSpeaker Directory Naming Pattern Examples:")
    for spk in list(speakers_with_files.keys())[:5]:
        print(f"  - Folder: '{spk}' ({len(speaker_file_map[spk])} wav files)")

    # Check if speaker folders contain roll numbers / names
    roll_num_speakers = [s for s in speakers_with_files if re.search(r'\d{5,}', s)]
    print(f"Speaker folders containing student roll numbers / explicit speaker IDs: {len(roll_num_speakers)} / {len(speakers_with_files)}")

    # 4. Audio Quality, Sampling Rates & Duration Statistics
    print("\n--- 3. AUDIO PROPERTIES & DURATION STATISTICS ---")
    sample_rates = Counter()
    channel_counts = Counter()
    durations = []
    corrupted_files = []

    for f in wav_files:
        try:
            with wave.open(str(f), 'rb') as wf:
                channels = wf.getnchannels()
                sr = wf.getframerate()
                frames = wf.getnframes()
                duration = frames / float(sr)

                channel_counts[channels] += 1
                sample_rates[sr] += 1
                durations.append(duration)
        except Exception as e:
            corrupted_files.append((f, str(e)))

    if sample_rates:
        print("Sampling Rates across WAV files:")
        for sr, count in sorted(sample_rates.items()):
            print(f"  - {sr:6d} Hz: {count:4d} files ({count/len(wav_files)*100:.1f}%)")

        print("\nChannel Configurations:")
        for ch, count in sorted(channel_counts.items()):
            ch_name = "Mono" if ch == 1 else ("Stereo" if ch == 2 else f"{ch}-channel")
            print(f"  - {ch_name:10s}: {count:4d} files ({count/len(wav_files)*100:.1f}%)")

        min_dur = min(durations)
        max_dur = max(durations)
        mean_dur = sum(durations) / len(durations)
        total_dur_sec = sum(durations)

        durations_sorted = sorted(durations)
        median_dur = durations_sorted[len(durations_sorted)//2]

        print("\nDuration Statistics:")
        print(f"  - Min duration:     {min_dur:.3f} seconds")
        print(f"  - Max duration:     {max_dur:.3f} seconds")
        print(f"  - Mean duration:    {mean_dur:.3f} seconds")
        print(f"  - Median duration:  {median_dur:.3f} seconds")
        print(f"  - Total audio time: {total_dur_sec / 60.0:.2f} minutes ({total_dur_sec / 3600.0:.2f} hours)")

    print("\n--- 4. CORRUPTED OR UNREADABLE FILES ---")
    if corrupted_files:
        print(f"Found {len(corrupted_files)} unreadable/corrupted files:")
        for bad_file, err in corrupted_files[:10]:
            print(f"  - {bad_file.name}: {err}")
    else:
        print("Zero corrupted files found! All WAV headers parsed cleanly.")

    print("\n" + "=" * 70)
    return True

if __name__ == "__main__":
    inspect_raw_dataset(os.path.join("data", "raw"))
