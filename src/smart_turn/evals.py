import argparse
from collections import defaultdict

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import SmartTurnDataset, load_manifest, train_val_split
from model_upd import SmartTurnModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(labels: list, probs: list) -> dict:
    n = len(labels)
    preds = [1.0 if p > 0.5 else 0.0 for p in probs]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)

    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)


    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else None

    return {"n": n, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


@torch.no_grad()
def run_eval(
    checkpoint_path: str = "checkpoints/best_model.pt",
    manifest_path: str = "data/manifest.csv",
    val_frac: float = 0.15,
    batch_size: int = 32,
    full: bool = False,
):
    
    device = get_device()

    rows = load_manifest(manifest_path)
    if full:
        eval_rows = rows
    else:
        _, eval_rows = train_val_split(rows, val_frac=val_frac)  # must match train.py's split (same defaults)

    ckpt = torch.load(checkpoint_path, map_location=device)
    unfreeze_top = ckpt.get("unfreeze_top", 0)
    model = SmartTurnModel(freeze_encoder=True, unfreeze_top=unfreeze_top).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # shuffle=False is critical - keeps prediction order aligned with eval_rows
    loader = DataLoader(SmartTurnDataset(eval_rows), batch_size=batch_size, shuffle=False)

    all_probs = []
    for feats, _labels in loader:
        feats = feats.to(device)
        logits, _ = model(feats)
        probs = torch.sigmoid(logits).cpu()
        all_probs.extend(probs.tolist())
    assert len(all_probs) == len(eval_rows), "prediction/row count mismatch - did shuffling break alignment?"

    groups = defaultdict(lambda: {"labels": [], "probs": []})

    def bump(key, label, prob):
        groups[key]["labels"].append(label)
        groups[key]["probs"].append(prob)

    for row, prob in zip(eval_rows, all_probs):
        label = float(row["endpoint_bool"].lower() == "true")
        bump("overall", label, prob)
        bump(f"lang={row['language']}", label, prob)
        bump(f"source={row['dataset']}", label, prob)
        bump(f"synthetic={row['synthetic']}", label, prob)
        bump(f"lang={row['language']}|source={row['dataset']}", label, prob)

    print(f"{'group':45s} {'n':>6} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'auc':>7}")
    print("-" * 98)
    for key in sorted(groups.keys()):
        m = compute_metrics(groups[key]["labels"], groups[key]["probs"])
        auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "n/a"
        # flag tiny groups - easy to over-read noise as signal on small n
        flag = "  (small n)" if m["n"] < 30 else ""
        print(f"{key:45s} {m['n']:6d} {m['accuracy']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} {auc_str:>7}{flag}")

    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--full", action="store_true", help="evaluate on the entire manifest, no train/val split (use for an independent test-set manifest)")
    args = parser.parse_args()
    run_eval(checkpoint_path=args.checkpoint, manifest_path=args.manifest, full=args.full)