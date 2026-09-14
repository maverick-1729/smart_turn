"""
Inference module for Smart Turn (Hinglish) endpoint detection.

Loads a trained checkpoint once and exposes a single `predict()` function
meant to be imported directly by a demo app, e.g.:

    from inference import predict
    result = predict(audio)  # -> {"probability": 0.94, "endpoint_bool": True}

Designed for a Gradio-style Hugging Face Space, where the audio input is
either a file path or a (sample_rate, np.ndarray) tuple (Gradio's default
Audio component output format).
"""

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch

from features import extract_features
from model_upd import SmartTurnModel

DEFAULT_CHECKPOINT = "checkpoints/ablation_fusion_unfrozen_only.pt"
_TMP_WAV_PATH = "/tmp/_smart_turn_inference_input.wav"

# Module-level cache so a demo app doesn't reload the checkpoint (and
# re-init WhisperTiny) on every single call.
_model = None
_model_checkpoint_path = None


def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT) -> SmartTurnModel:
    """
    Reconstructs SmartTurnModel from a checkpoint.

    New checkpoints store architecture kwargs explicitly.  Older checkpoints
    only store ``model_state_dict`` and ``unfreeze_top``; for those, infer the
    fusion width from ``fusion.raw_weights``.  This is the authoritative
    source for the number of fused encoder states and avoids constructing a
    two-layer fusion module for a three-layer checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    unfreeze_top = ckpt.get("unfreeze_top", 0)
    saved_fusion_weights = state_dict.get("fusion.raw_weights")
    inferred_num_fusion_layers = (
        saved_fusion_weights.numel() if saved_fusion_weights is not None else None
    )
    num_fusion_layers = ckpt.get("num_fusion_layers", inferred_num_fusion_layers)
    if num_fusion_layers is None:
        raise KeyError(
            "Checkpoint has neither 'num_fusion_layers' metadata nor "
            "'fusion.raw_weights' in model_state_dict."
        )

    model = SmartTurnModel(
        freeze_encoder=True,
        unfreeze_top=unfreeze_top,
        num_fusion_layers=num_fusion_layers,
        position_embed_mode=ckpt.get("position_embed_mode", "truncate"),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model(checkpoint_path: str = DEFAULT_CHECKPOINT) -> SmartTurnModel:
    """Cached singleton accessor - reloads only if checkpoint_path changes."""
    global _model, _model_checkpoint_path
    if _model is None or _model_checkpoint_path != checkpoint_path:
        _model = load_model(checkpoint_path)
        _model_checkpoint_path = checkpoint_path
    return _model


def _to_wav_path(audio: Union[str, Path, tuple, np.ndarray]) -> str:
    """
    Normalizes demo-app audio input into a wav file path, since
    extract_features() (per export_onnx.py) operates on file paths.

    Handles:
      - str / Path: already a file path, passed straight through.
      - (sample_rate, samples) tuple: Gradio's default Audio component
        output. Written to a temp wav via soundfile.
    """
    if isinstance(audio, (str, Path)):
        return str(audio)

    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, samples = audio
        samples = np.asarray(samples)
        if np.issubdtype(samples.dtype, np.integer):
            # int16 PCM from Gradio -> float32 in [-1, 1]
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        else:
            samples = samples.astype(np.float32)
        sf.write(_TMP_WAV_PATH, samples, sample_rate)
        return _TMP_WAV_PATH

    raise TypeError(
        f"Unsupported audio input type: {type(audio)}. "
        "Expected a file path (str/Path) or a (sample_rate, samples) tuple."
    )


def predict(
    audio: Union[str, Path, tuple, np.ndarray],
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> float:
    """
    Runs one audio clip through the model.

    Args:
        audio: wav file path, or (sample_rate, np.ndarray) tuple.
        checkpoint_path: path to a .pt checkpoint from train.py.

    Returns:
        probability (float): P(speaker has finished their turn). No
        threshold is applied here - callers decide what to do with the
        raw score (e.g. a live demo just displays it).
    """
    model = get_model(checkpoint_path)
    wav_path = _to_wav_path(audio)

    feats = extract_features(wav_path).astype(np.float32)
    x = torch.from_numpy(feats).unsqueeze(0)  # (1, 80, N_MEL_FRAMES)

    with torch.no_grad():
        logits, _attn_weights = model(x)
        probability = torch.sigmoid(logits).item()

    return probability


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run smart-turn inference on a single wav file")
    parser.add_argument("wav_path", help="Path to a wav file")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    probability = predict(args.wav_path, checkpoint_path=args.checkpoint)
    print(f"probability={probability:.4f}")
