# Results

## Fusion vs. No Fusion

Effect of ELMo-style layer fusion (combining multiple WhisperTiny encoder
layers) vs. using only the final encoder layer.

| Variant                  | AUC | Accuracy | Inference latency (CPU, ms) |
|---------------------------|-----|---------|------------------------------|
| Without fusion (last layer only) | 0.97 | 0.92 | ~13 |
| With fusion (`num_fusion_layers=3`) | 0.98 | 0.94 | ~13 |

**Takeaway:** — Fusion gives better performance with almost similar latency indicating output from previous
encoder layers gives meaningful signal.

## Truncating vs. Interpolating Embeddings

Comparison of 2 strategies regarding embeddings of whisper - truncating (to required frames)
vs. interpolating them (after the convolution layers before encoder) to the required frames.

| Variant | Accuracy | 
|---------|-----|
| Truncating    | 0.94 | 
| Interpolating | 0.86 | 

**Takeaway:** Interpolating severely degrades the performance possibly because encoder layers (which are pre-trained)
are expecting the input to follow a distribution and interpolating them leads to inputs being different from the 
distribution. 

## Model Size

| Format | Size |
|--------|------|
| PyTorch checkpoint (fp32) | 32 MB |
| ONNX (fp32) | 32 MB |
| ONNX (int8, dynamically quantized) | 8 MB |

## Latency (CPU)

| Format | Latency (ms) (single threaded)| 
|--------|--------------------|
| PyTorch (fp32) | ~13 | 
| ONNX (fp32) | ~20 | 
| ONNX (int8) | ~20~ |

## Final Conclusion

Our Fusion (ELMo style) model gives on par performance with the official 
pipecat model for Hindi and English languages (both at ~94% accuracy).
In terms of latency, on similar hardware, our model has a latency of ~13ms while the pipecat model
tested locally gives a latency of ~35ms. 
Our model has almost one-third latency compared to the official model.
