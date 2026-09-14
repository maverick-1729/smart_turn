import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

from features import MAX_AUDIO_SECONDS, N_MEL_FRAMES, WHISPER_MODEL_NAME

ENCODER_SEQ_LEN = N_MEL_FRAMES // 2  # Whisper encoder's stride-2 conv halves time
NUM_FUSION_LAYERS = 3  # trailing encoder hidden states to fuse (incl. final layer)


class AttentionPool(nn.Module):


    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        weights = torch.softmax(self.score(x), dim=1)  # (batch, seq_len, 1)
        pooled = (weights * x).sum(dim=1)  # (batch, d_model)
        return pooled, weights.squeeze(-1)


class LayerFusion(nn.Module):
    """
    ELMo-style scalar mixing of the last `num_layers` encoder hidden states.
    Learns a softmax-normalized weight per layer plus an overall scale
    (gamma), so the model decides how much each trailing layer contributes
    to the pooled representation instead of only using the final layer.

    """

    def __init__(self, num_layers: int, warm_start_last: bool = True):
        super().__init__()
        if warm_start_last:
            # Bias the initial softmax toward the final layer so training
            # starts equivalent to "last_hidden_state only" (matching the
            # pre-fusion baseline's representation quality from step one),
            # and the model only pulls weight toward earlier layers if that
            # measurably helps - instead of starting from a uniform average
            # that dilutes the well-tuned final layer before anything has
            # had a chance to adapt. softmax([0,0,4]) puts >95% on the last.
            init = torch.zeros(num_layers)
            init[-1] = 4.0
        else:
            init = torch.zeros(num_layers)
        self.raw_weights = nn.Parameter(init)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        # hidden_states: list of (batch, seq_len, d_model), length == num_layers
        weights = torch.softmax(self.raw_weights, dim=0)
        fused = sum(w * h for w, h in zip(weights, hidden_states))
        return self.gamma * fused


def _resize_positional_embeddings(old_weight: torch.Tensor, new_len: int, mode: str) -> torch.Tensor:
    """
    Resize a (old_len, d_model) positional embedding table down to
    (new_len, d_model), either by:
    - "truncate": keeping the first new_len rows as-is (the original
      approach - exact pretrained values, but discards the embedding
      table's positional structure beyond an arbitrary early cutoff).
    - "interpolate": linear interpolation along the position axis (resizes
      the whole table down instead of slicing it - preserves relative
      spacing, but resamples every value, which is a real distributional
      shift the encoder was never trained on).

    """
    if mode == "truncate":
        return old_weight[:new_len].clone()
    if mode == "interpolate":
        x = old_weight.T.unsqueeze(0)  # (1, d_model, old_len)
        x = F.interpolate(x, size=new_len, mode="linear", align_corners=True)
        return x.squeeze(0).T.contiguous()  # (new_len, d_model)
    raise ValueError(f"Unknown position_embed_mode: {mode!r} (expected 'truncate' or 'interpolate')")


class SmartTurnModel(nn.Module):
    def __init__(
        self,
        whisper_model_name: str = WHISPER_MODEL_NAME,
        freeze_encoder: bool = True,
        unfreeze_top: int = 0,
        num_fusion_layers: int = NUM_FUSION_LAYERS,
        position_embed_mode: str = "truncate",
    ):
        super().__init__()

        full = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = full.encoder
        d_model = full.config.d_model

        old_pos_weight = self.encoder.embed_positions.weight.data
        assert old_pos_weight.shape[0] >= ENCODER_SEQ_LEN, (
            f"Requested encoder_seq_len={ENCODER_SEQ_LEN} exceeds pretrained "
            f"positional embedding size {old_pos_weight.shape[0]}"
        )
        new_pos = nn.Embedding(ENCODER_SEQ_LEN, d_model)
        new_pos.weight.data = _resize_positional_embeddings(old_pos_weight, ENCODER_SEQ_LEN, position_embed_mode)
        self.encoder.embed_positions = new_pos
        self.position_embed_mode = position_embed_mode

        self.encoder.config.max_source_positions = ENCODER_SEQ_LEN

        self.freeze_encoder = freeze_encoder
        self.unfreeze_top = unfreeze_top
        self._apply_freeze(freeze_encoder, unfreeze_top)

        n_layers = len(self.encoder.layers)
        assert 1 <= num_fusion_layers <= n_layers + 1, (
            f"num_fusion_layers={num_fusion_layers} out of range for "
            f"{n_layers}-layer encoder (max {n_layers + 1}, incl. embedding output)"
        )
        self.num_fusion_layers = num_fusion_layers
        self.fusion = LayerFusion(num_fusion_layers)

        self.pool = AttentionPool(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def _apply_freeze(self, freeze_encoder: bool, unfreeze_top: int):
        n_layers = len(self.encoder.layers)
        assert 0 <= unfreeze_top <= n_layers, (
            f"unfreeze_top={unfreeze_top} out of range for {n_layers}-layer encoder"
        )

        if not freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            self._fully_frozen = False
            return

        # Start fully frozen, including conv1/conv2/embed_positions...
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        if unfreeze_top > 0:
            for layer in self.encoder.layers[-unfreeze_top:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for p in self.encoder.layer_norm.parameters():
                p.requires_grad_(True)

        self._fully_frozen = (unfreeze_top == 0)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder and self._fully_frozen:
            self.encoder.eval()
        return self

    def forward(self, input_features: torch.Tensor):
        with torch.set_grad_enabled(not self._fully_frozen):
            enc_out = self.encoder(input_features, output_hidden_states=True)

        hidden_states = list(enc_out.hidden_states[-self.num_fusion_layers:])
        fused = self.fusion(hidden_states)  # (B, T, D)

        pooled, attn_weights = self.pool(fused)
        logits = self.head(pooled).squeeze(-1)  # (B,)
        return logits, attn_weights


if __name__ == "__main__":
    for label, kwargs in [
        ("fully frozen", dict(freeze_encoder=True, unfreeze_top=0)),
        ("top-1 layer unfrozen", dict(freeze_encoder=True, unfreeze_top=1)),
        ("fully unfrozen", dict(freeze_encoder=False)),
    ]:
        model = SmartTurnModel(**kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{label}] total: {n_params:,} | trainable: {n_trainable:,}")

    dummy = torch.randn(4, 80, N_MEL_FRAMES)  # batch of 4
    logits, attn = model(dummy)
    print(f"logits shape: {logits.shape}")  # (4,)
    print(f"attn weights shape: {attn.shape}")  # (4, ENCODER_SEQ_LEN)
    print(f"sample probs: {torch.sigmoid(logits)}")