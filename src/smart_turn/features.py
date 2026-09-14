import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

WHISPER_MODEL_NAME = "openai/whisper-tiny"
TARGET_SR = 16000

# Cap audio length fed to the encoder. Shorter = faster on CPU, since
# attention cost scales with sequence length. Adjust if your endpointing
# clips are commonly longer than this and you're seeing truncation hurt
# accuracy - trade-off is latency vs. context.
MAX_AUDIO_SECONDS = 8.0
N_MEL_FRAMES = int(MAX_AUDIO_SECONDS * 100)  # 100 frames/sec at Whisper's 10ms hop


def _get_feature_extractor() -> WhisperFeatureExtractor:
    # chunk_length controls the fixed padding/truncation length (in seconds)
    # that the extractor pads/trims every clip to - override the Whisper
    # default of 30 down to MAX_AUDIO_SECONDS.
    return WhisperFeatureExtractor.from_pretrained(
        WHISPER_MODEL_NAME,
        chunk_length=MAX_AUDIO_SECONDS,
    )


_feature_extractor: WhisperFeatureExtractor = None  # lazy singleton


def get_feature_extractor() -> WhisperFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = _get_feature_extractor()
    return _feature_extractor


def load_audio(wav_path: str) -> np.ndarray:
    """Load a wav file, resampling to 16kHz mono if needed."""
    array, sr = sf.read(wav_path, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
    return array


def right_align(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Align audio so the END of the clip lands at the END of a fixed-length
    array - since for turn detection, the signal that matters (did the
    speaker just finish) lives at the end of the clip, not the start.

    - If audio is longer than n_samples: keep the LAST n_samples (drop
      older audio, not the recent audio near the endpoint).
    - If audio is shorter: left-pad with silence (zeros), so the real
      audio still ends exactly at the last frame.
    """
    if len(audio) >= n_samples:
        return audio[-n_samples:]
    pad_len = n_samples - len(audio)
    return np.concatenate([np.zeros(pad_len, dtype=audio.dtype), audio])


def extract_features(wav_path: str) -> np.ndarray:
    """
    Load a wav file and return its log-Mel spectrogram, shaped
    (n_mels=80, N_MEL_FRAMES), matching WhisperTiny's expected encoder input
    (truncated in time vs. the default 30s/1500 frames, and right-aligned
    so the clip's end - the relevant part for endpoint detection - always
    lands at the last frame rather than being truncated away or diluted by
    trailing zero-padding).
    """
    audio = load_audio(wav_path)
    audio = right_align(audio, n_samples=int(MAX_AUDIO_SECONDS * TARGET_SR))

    fe = get_feature_extractor()
    features = fe(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="np",
        padding="do_not_pad",  # already exact length, don't let the
                                # extractor re-pad/truncate from the front
    )
    # shape: (1, n_mels, N_MEL_FRAMES) -> drop batch dim
    return features["input_features"][0]


if __name__ == "__main__":
    import csv
    import sys

    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/manifest.csv"
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    print(f"Manifest has {len(rows)} rows. Testing feature extraction on first 3...")
    for row in rows[:3]:
        feats = extract_features(row["wav_path"])
        print(f"  {row['id']} ({row['language']}, endpoint_bool={row['endpoint_bool']}): "
              f"features shape = {feats.shape}")