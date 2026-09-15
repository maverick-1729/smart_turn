---
title: Smart Turn Hinglish
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
---

## Overview

A turn-detection model for Hinglish conversational speech — it
predicts whether a speaker has finished their conversational turn, given
audio ending at the current moment. It's built for low-latency inference on
CPU (no GPU required at inference time), targeting real-time voice
applications where you need to decide *right now* whether to let the model
keep listening or hand control back to the system.

[live demo](https://huggingface.co/spaces/maverick1729/smart_turn)

## Model Input

- **Format:** raw audio waveform, mono, resampled to 16kHz (Whisper's expected
  sample rate).
- **Window:** a fixed-length window of the trailing **8 seconds** of audio,
  right-aligned — i.e. if a clip is shorter than 8s, it's padded; if longer,
  only the most recent 8 seconds are used. This matches how turn-taking
  signal is concentrated near the end of an utterance.
- **Features:** the raw waveform is converted to an 80-channel log-mel
  spectrogram via `WhisperFeatureExtractor` before being passed to the model,
  giving a fixed-shape tensor of `(80, N_MEL_FRAMES)` per input.

## Model Output

- A single scalar **probability** in `[0, 1]`, produced via a sigmoid over
  the model's output logit.
- This represents **P(speaker has finished their turn)** as of the end of
  the input window — not a binary decision. No threshold is baked into the
  model; callers choose their own operating point (or, as in this repo's
  demo app, just display the raw probability).

## Architecture

```
waveform (16kHz, mono)
        │
        ▼
WhisperFeatureExtractor  →  log-mel spectrogram (80, N_MEL_FRAMES)
        │
        ▼
WhisperTiny encoder (whisper-tiny, 4 layers, 384-dim hidden state)
  - last 2 layers of encoder are fine-tuned
        │
        ▼
Layer Fusion (ELMo-style)
  - learned per-layer weights + a scalar gamma combine hidden states
    from `num_fusion_layers` encoder layers into a single sequence of
    frame-level representations, rather than using only the final
    encoder layer
        │
        ▼
Attention pooling over time
  - the model returns per-frame attention weights alongside the
    prediction, used to pool the fused sequence into a single
    utterance-level vector.
        │
        ▼
Classification head → 1 logit
        │
        ▼
sigmoid → P(turn finished)
```

**Design rationale:**
- The **frozen WhisperTiny encoder** keeps the parameter count (and
  therefore CPU inference latency) small, while still benefiting from
  Whisper's pretrained multilingual audio representations — useful given
  Hinglish's code-switching between Hindi and English.
- **Layer fusion** lets the model draw on representations from multiple
  encoder depths rather than committing to just the last layer, which
  matters since prosodic/turn-taking cues and phonetic/lexical cues can
  live at different depths of the encoder.

## Inference & Deployment

- Trained checkpoints (`.pt`) are exported to ONNX and dynamically
  quantized to int8 for further CPU latency reduction (`export_onnx.py`),
  following the same approach as the official Smart Turn v3 model.
- A Gradio demo app (`app.py`) wraps the model for live microphone input,
  re-running inference on the trailing 8-second window as audio streams in.