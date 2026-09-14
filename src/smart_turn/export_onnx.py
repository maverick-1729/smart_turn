import argparse

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from dataset import load_manifest
from features import N_MEL_FRAMES, extract_features
from model_upd import SmartTurnModel


class InferenceWrapper(nn.Module):

    def __init__(self, model: SmartTurnModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return torch.sigmoid(logits)


def load_torch_model(checkpoint_path: str = "checkpoints/ablation_fusion_unfrozen_only.pt") -> SmartTurnModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    unfreeze_top = ckpt.get("unfreeze_top", 0)
    num_fusion_layers = ckpt.get("num_fusion_layers", 3)  
    model = SmartTurnModel(
        freeze_encoder=True,
        unfreeze_top=unfreeze_top,
        num_fusion_layers=num_fusion_layers,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def export_onnx(
    checkpoint_path: str = "checkpoints/best_model.pt",
    onnx_path: str = "checkpoints/model_fp32.onnx",
    opset: int = 17,
) -> str:
    model = load_torch_model(checkpoint_path)
    wrapper = InferenceWrapper(model).eval()

    dummy = torch.randn(1, 80, N_MEL_FRAMES)

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input_features"],
        output_names=["probability"],
        opset_version=opset,
    )
    print(f"Exported fp32 ONNX (static batch=1) to {onnx_path}")
    return onnx_path


def quantize_onnx(
    onnx_fp32_path: str = "checkpoints/model_fp32.onnx",
    onnx_int8_path: str = "checkpoints/model_int8.onnx",
) -> str:

    from onnxruntime.quantization.shape_inference import quant_pre_process

    preprocessed_path = onnx_fp32_path.replace(".onnx", "_preprocessed.onnx")
    quant_pre_process(onnx_fp32_path, preprocessed_path, skip_symbolic_shape=False)
    print(f"Preprocessed (symbolic shape inference) ONNX to {preprocessed_path}")

    quantize_dynamic(preprocessed_path, onnx_int8_path, weight_type=QuantType.QInt8)
    print(f"Quantized int8 ONNX saved to {onnx_int8_path}")
    return onnx_int8_path


def verify_parity(
    checkpoint_path: str,
    onnx_path: str,
    manifest_path: str = "data/manifest.csv",
    n: int = 30,
) -> float:

    torch_model = load_torch_model(checkpoint_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    rows = load_manifest(manifest_path)[:n]
    diffs = []
    flipped = 0
    for row in rows:
        feats = extract_features(row["wav_path"]).astype(np.float32)
        x = torch.from_numpy(feats).unsqueeze(0)
        with torch.no_grad():
            logits, _ = torch_model(x)
            torch_prob = torch.sigmoid(logits).item()
        onnx_prob = float(sess.run(None, {"input_features": feats[None]})[0].item())
        diffs.append(abs(torch_prob - onnx_prob))
        if (torch_prob > 0.5) != (onnx_prob > 0.5):
            flipped += 1

    max_diff = max(diffs)
    print(f"  max |prob diff| over {n} samples: {max_diff:.6f} | predictions flipped: {flipped}/{n}")
    return max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ablation_fusion_unfrozen_only.pt")
    parser.add_argument("--fp32-out", default="checkpoints/ablation_fusion_unfrozen_only_fp32.onnx")
    parser.add_argument("--int8-out", default="checkpoints/ablation_fusion_unfrozen_only_int8.onnx")
    parser.add_argument("--manifest", default="data/manifest.csv")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.fp32_out)
    quantize_onnx(args.fp32_out, args.int8_out)

    print("\nVerifying fp32 ONNX matches PyTorch (expect near-zero diff):")
    verify_parity(args.checkpoint, args.fp32_out, args.manifest)

    print("\nVerifying int8 ONNX matches PyTorch (expect a small but nonzero diff):")
    verify_parity(args.checkpoint, args.int8_out, args.manifest)