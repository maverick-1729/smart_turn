import argparse
import csv
import hashlib
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperFeatureExtractor


MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
    "smart-turn-v3.2-cpu.onnx?download=true"
)
MODEL_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
DEFAULT_MODEL_PATH = Path("checkpoints/smart-turn-v3.2-cpu.onnx")
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 8 * SAMPLE_RATE
FEATURE_SHAPE = (1, 80, 800)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(path: Path) -> None:
    """Download the published CPU checkpoint and verify its official hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    print(f"Downloading official Smart Turn v3.2 CPU model to {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        actual_hash = sha256(temporary_path)
        if actual_hash != MODEL_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}."
            )
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print("Download verified.")


def load_audio_for_baseline(wav_path: str) -> np.ndarray:
    """Apply the official v3.2 waveform contract before feature extraction."""
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    # Pipecat's audio_utils.truncate_audio_to_last_n_seconds: retain the
    # endpoint, then left-pad shorter turns with silence.
    if len(audio) > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:]
    elif len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (WINDOW_SAMPLES - len(audio), 0))
    return np.asarray(audio, dtype=np.float32)


def make_features(extractor: WhisperFeatureExtractor, audio: np.ndarray) -> np.ndarray:
    features = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=WINDOW_SAMPLES,
        truncation=True,
        do_normalize=True,
    )["input_features"].astype(np.float32)
    if features.shape != FEATURE_SHAPE:
        raise ValueError(f"Expected features shaped {FEATURE_SHAPE}, got {features.shape}")
    return features


def percentile_stats(times_s: list[float]) -> str:
    values = np.asarray(times_s) * 1000
    return (
        f"mean={values.mean():7.2f} ms  p50={np.percentile(values, 50):7.2f} ms  "
        f"p95={np.percentile(values, 95):7.2f} ms  min={values.min():7.2f} ms  "
        f"max={values.max():7.2f} ms"
    )


def build_session(model_path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def benchmark(model_path: Path, manifest_path: Path, n_samples: int, warmup: int, threads: int) -> None:
    with manifest_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    if n_samples > 0:
        rows = rows[:n_samples]

    extractor = WhisperFeatureExtractor(chunk_length=8)
    session = build_session(model_path, threads)
    input_name = session.get_inputs()[0].name
    print(
        f"Official model: {model_path}\n"
        f"Manifest: {manifest_path} ({len(rows)} clips)\n"
        f"Machine: {platform.platform()} | Python {platform.python_version()} | "
        f"ONNX Runtime {ort.__version__}\n"
        f"Provider: {session.get_providers()[0]} | intra-op threads: {threads} | "
        "inter-op threads: 1 | batch size: 1"
    )

    # Decode and feature-extract once for direct model latency.  Keeping this
    # separate from end-to-end latency lets comparisons distinguish model cost
    # from the common audio/preprocessing cost.
    inputs = [make_features(extractor, load_audio_for_baseline(row["wav_path"])) for row in rows]
    for features in inputs[:warmup]:
        session.run(None, {input_name: features})

    model_times = []
    probabilities = []
    for features in inputs:
        started = time.perf_counter()
        output = session.run(None, {input_name: features})
        model_times.append(time.perf_counter() - started)
        probabilities.append(float(np.asarray(output[0]).reshape(-1)[0]))

    # This includes WAV reading, resampling (if needed), the official Whisper
    # feature extractor, and the ONNX forward pass for each independent clip.
    for row in rows[:warmup]:
        features = make_features(extractor, load_audio_for_baseline(row["wav_path"]))
        session.run(None, {input_name: features})
    end_to_end_times = []
    for row in rows:
        started = time.perf_counter()
        features = make_features(extractor, load_audio_for_baseline(row["wav_path"]))
        session.run(None, {input_name: features})
        end_to_end_times.append(time.perf_counter() - started)

    print(f"\nLatency over {len(rows)} clips (after {warmup} warmup runs):")
    print(f"  model_forward: {percentile_stats(model_times)}")
    print(f"  end_to_end:   {percentile_stats(end_to_end_times)}")
    print(f"  output sanity check: first probability={probabilities[0]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_test.csv"))
    parser.add_argument("--n-samples", type=int, default=200, help="0 means all manifest rows")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--download", action="store_true", help="download and hash-verify the official model if missing")
    args = parser.parse_args()

    if args.num_threads < 1:
        parser.error("--num-threads must be at least 1")
    if args.n_samples < 0 or args.warmup < 0:
        parser.error("--n-samples and --warmup cannot be negative")
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    if not args.model.is_file():
        if args.download:
            download_model(args.model)
        else:
            parser.error(
                f"official model not found: {args.model}. Run again with --download."
            )
    if sha256(args.model) != MODEL_SHA256:
        parser.error(f"model SHA-256 does not match the official v3.2 CPU release: {args.model}")

    benchmark(args.model, args.manifest, args.n_samples, args.warmup, args.num_threads)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Benchmark interrupted.")
