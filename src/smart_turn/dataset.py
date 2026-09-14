import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features import extract_features

MANIFEST_PATH = "data/manifest.csv"


def load_manifest(manifest_path: str = MANIFEST_PATH) -> list[dict]:
    with open(manifest_path) as f:
        return list(csv.DictReader(f))


def train_val_split(rows: list[dict], val_frac: float = 0.15, seed: int = 42):
    """
    Stratified by (language, endpoint_bool) so the val set has a
    representative mix of both languages and both classes, not just
    whatever falls out of a random split.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["language"], row["endpoint_bool"])
        groups.setdefault(key, []).append(row)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for key, group_rows in groups.items():
        rng.shuffle(group_rows)
        n_val = max(1, int(len(group_rows) * val_frac))
        val_rows.extend(group_rows[:n_val])
        train_rows.extend(group_rows[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


class SmartTurnDataset(Dataset):
    """
    Reads (features, label) pairs from manifest rows. Features are cached to
    a .npy file next to the source .wav on first access, so re-running
    training (or later epochs) skips re-extraction.
    """

    def __init__(self, rows: list[dict], use_cache: bool = True):
        self.rows = rows
        self.use_cache = use_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wav_path = row["wav_path"]
        cache_path = Path(wav_path).with_suffix(".npy")

        if self.use_cache and cache_path.exists():
            feats = np.load(cache_path)
        else:
            feats = extract_features(wav_path)
            if self.use_cache:
                np.save(cache_path, feats)

        label = float(row["endpoint_bool"].lower() == "true") if isinstance(row["endpoint_bool"], str) else float(row["endpoint_bool"])
        return torch.from_numpy(feats).float(), torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    rows = load_manifest()
    train_rows, val_rows = train_val_split(rows)
    print(f"Total: {len(rows)} | Train: {len(train_rows)} | Val: {len(val_rows)}")

    from collections import Counter
    print("Train language/label mix:", Counter((r["language"], r["endpoint_bool"]) for r in train_rows))
    print("Val language/label mix:", Counter((r["language"], r["endpoint_bool"]) for r in val_rows))

    ds = SmartTurnDataset(train_rows)
    feats, label = ds[0]
    print(f"Sample features shape: {feats.shape}, label: {label}")