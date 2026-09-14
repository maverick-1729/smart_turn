

import argparse
import time

import numpy as np
import torch

from dataset import load_manifest, train_val_split
from features import extract_features
from model_upd import SmartTurnModel

DEVICE = torch.device("cpu")  # forced


def load_model(checkpoint_path: str = "checkpoints/ablation_fusion_unfrozen_only.pt") -> SmartTurnModel:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    unfreeze_top = ckpt.get("unfreeze_top", 0)
    model = SmartTurnModel(freeze_encoder=True, unfreeze_top=unfreeze_top).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_single_torch(model: SmartTurnModel, wav_path: str):
    t0 = time.perf_counter()
    feats = extract_features(wav_path)  # (80, N_MEL_FRAMES) numpy
    t1 = time.perf_counter()

    x = torch.from_numpy(feats).float().unsqueeze(0)  # (1, 80, N_MEL_FRAMES)
    logits, _ = model(x)
    prob = torch.sigmoid(logits).item()
    t2 = time.perf_counter()

    return prob, (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def predict_single_onnx(session, wav_path: str):
    t0 = time.perf_counter()
    feats = extract_features(wav_path).astype(np.float32)  # (80, N_MEL_FRAMES)
    t1 = time.perf_counter()

    prob = float(session.run(None, {"input_features": feats[None]})[0].item())
    t2 = time.perf_counter()

    return prob, (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def print_stats(name: str, times_ms: list):
    arr = np.array(times_ms)
    print(
        f"{name:16s} mean={arr.mean():7.2f}ms  median={np.median(arr):7.2f}ms  "
        f"p95={np.percentile(arr, 95):7.2f}ms  min={arr.min():7.2f}ms  max={arr.max():7.2f}ms"
    )


def benchmark(
    checkpoint_path: str = "checkpoints/best_model.pt",
    manifest_path: str = "data/manifest.csv",
    n_samples: int = 200,
    warmup: int = 10,
    val_frac: float = 0.15,
    num_threads: int = None,
    verbose: bool = False,
    full: bool = False,
    backend: str = "torch",
    onnx_path: str = None,
):
    """
    Args:
        backend: "torch" (original PyTorch model), "onnx" (fp32 ONNX
            export), or "onnx-int8" (quantized ONNX) - see export_onnx.py
            to produce the .onnx files. Lets you compare latency across
            all three on identical inputs/harness.
        onnx_path: required (and only used) when backend is "onnx" or
            "onnx-int8" - path to the corresponding .onnx file.
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    print(f"Device: {DEVICE} | torch threads: {torch.get_num_threads()} | backend: {backend}")

    if backend == "torch":
        model = load_model(checkpoint_path)
        predict_fn = lambda wav_path: predict_single_torch(model, wav_path)
    elif backend in ("onnx", "onnx-int8"):
        import onnxruntime as ort
        assert onnx_path is not None, "--onnx-path is required for onnx/onnx-int8 backends"
        sess_options = ort.SessionOptions()
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
        session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        predict_fn = lambda wav_path: predict_single_onnx(session, wav_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    rows = load_manifest(manifest_path)
    if not full:
        _, rows = train_val_split(rows, val_frac=val_frac)  # benchmark on held-out data, not train
    rows = rows[: warmup + n_samples]

    # First few CPU forward passes are typically slower (thread pool spin-up,
    # lazy kernel/allocator init) - warm up and discard before measuring.
    for row in rows[:warmup]:
        predict_fn(row["wav_path"])

    feat_times, model_times, total_times = [], [], []
    for i, row in enumerate(rows[warmup: warmup + n_samples]):
        prob, ft, mt, tt = predict_fn(row["wav_path"])
        feat_times.append(ft)
        model_times.append(mt)
        total_times.append(tt)
        if verbose:
            print(f"  [{i}] {row['id']} ({row['language']}): prob={prob:.3f} total={tt:.2f}ms")

    print(f"\nLatency over {len(total_times)} examples (after {warmup} warmup runs, CPU, batch size 1, backend={backend}):")
    print_stats("feature_extract", feat_times)
    print_stats("model_forward", model_times)
    print_stats("total", total_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=None, help="e.g. 1 to test single-threaded worst case")
    parser.add_argument("--verbose", action="store_true", help="print per-example predictions and latency")
    parser.add_argument("--full", action="store_true", help="use the entire manifest, no train/val split (for an independent test-set manifest)")
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx", "onnx-int8"])
    parser.add_argument("--onnx-path", default=None, help="required for onnx/onnx-int8 backends")
    args = parser.parse_args()

    benchmark(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        n_samples=args.n_samples,
        warmup=args.warmup,
        num_threads=args.num_threads,
        verbose=args.verbose,
        full=args.full,
        backend=args.backend,
        onnx_path=args.onnx_path,
    )