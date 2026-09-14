import argparse
import csv
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SmartTurnDataset, load_manifest, train_val_split
from model_upd import SmartTurnModel

RUNS_LOG_PATH = "runs/results.csv"     
RUNS_HISTORY_DIR = "runs/history"      


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, desc="val"):
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = tn = fn = 0

    for feats, labels in tqdm(loader, desc=desc, leave=False):
        feats, labels = feats.to(device), labels.to(device)
        logits, _ = model(feats)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(labels)
        n += len(labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def log_run(run_id: str, config: dict, epoch_history: list, best_epoch: int, best_metrics: dict):
    os.makedirs(RUNS_HISTORY_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RUNS_LOG_PATH), exist_ok=True)

    history_path = os.path.join(RUNS_HISTORY_DIR, f"{run_id}.json")
    with open(history_path, "w") as f:
        json.dump({"run_id": run_id, "config": config, "epoch_history": epoch_history}, f, indent=2)

    summary_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **config,
        "best_epoch": best_epoch,
        "best_val_loss": best_metrics["loss"],
        "val_accuracy": best_metrics["accuracy"],
        "val_precision": best_metrics["precision"],
        "val_recall": best_metrics["recall"],
        "val_f1": best_metrics["f1"],
        "history_path": history_path,
    }

    file_exists = os.path.exists(RUNS_LOG_PATH)
    with open(RUNS_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_row)

    print(f"Logged run to {RUNS_LOG_PATH} (history: {history_path})")


def train(
    manifest_path: str = "data/manifest.csv",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    encoder_lr: float = 1e-5,
    weight_decay: float = 0.01,
    val_frac: float = 0.15,
    unfreeze_top: int = 0,
    checkpoint_path: str = "checkpoints/best_model_upd.pt",
    num_workers: int = 2,
    patience: int = 3,
    num_fusion_layers: int = 3,
    position_embed_mode: str = "truncate",
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "manifest_path": manifest_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "encoder_lr": encoder_lr,
        "weight_decay": weight_decay,
        "val_frac": val_frac,
        "unfreeze_top": unfreeze_top,
        "checkpoint_path": checkpoint_path,
        "patience": patience,
        "num_fusion_layers": num_fusion_layers,
        "position_embed_mode": position_embed_mode,
    }

    device = get_device()
    print(f"Using device: {device}")

    rows = load_manifest(manifest_path)
    train_rows, val_rows = train_val_split(rows, val_frac=val_frac)
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

    train_loader = DataLoader(
        SmartTurnDataset(train_rows), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        SmartTurnDataset(val_rows), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = SmartTurnModel(
        freeze_encoder=True,
        unfreeze_top=unfreeze_top,
        num_fusion_layers=num_fusion_layers,
        position_embed_mode=position_embed_mode,
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,}")

    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def is_encoder(name):
        return name.startswith("encoder.")

    def no_decay(name, param):
        return param.ndim <= 1  # biases, LayerNorm weight & bias

    groups = {
        "encoder_decay": [],
        "encoder_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    for n, p in named_trainable:
        bucket = "encoder" if is_encoder(n) else "head"
        bucket += "_no_decay" if no_decay(n, p) else "_decay"
        groups[bucket].append(p)

    for bucket, params in groups.items():
        n_params = sum(p.numel() for p in params)
        applied_lr = encoder_lr if bucket.startswith("encoder") else lr
        applied_wd = 0.0 if bucket.endswith("no_decay") else weight_decay
        print(f"  {bucket}: {n_params:,} params @ lr={applied_lr}, weight_decay={applied_wd}")

    optimizer = torch.optim.AdamW(
        [
            {"params": groups["head_decay"], "lr": lr, "weight_decay": weight_decay},
            {"params": groups["head_no_decay"], "lr": lr, "weight_decay": 0.0},
            {"params": groups["encoder_decay"], "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": groups["encoder_no_decay"], "lr": encoder_lr, "weight_decay": 0.0},
        ],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = None
    best_metrics = None
    epoch_history = []
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for feats, labels in pbar:
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            n_seen += len(labels)
            pbar.set_postfix(loss=f"{running_loss / n_seen:.4f}")

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:2d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | {elapsed:.1f}s"
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_s": elapsed,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                    "unfreeze_top": unfreeze_top,
                    "num_fusion_layers": num_fusion_layers,
                    "position_embed_mode": position_embed_mode,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint (val_loss={best_val_loss:.4f}) to {checkpoint_path}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(
                    f"\nEarly stopping: val_loss hasn't improved in {patience} epochs "
                    f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f})"
                )
                break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}, checkpoint: {checkpoint_path}")
    log_run(run_id, config, epoch_history, best_epoch, best_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--unfreeze-top", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/best_model_upd.pt")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-fusion-layers", type=int, default=3)
    parser.add_argument("--position-embed-mode", choices=["truncate", "interpolate"], default="truncate")
    args = parser.parse_args()

    train(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        unfreeze_top=args.unfreeze_top,
        checkpoint_path=args.checkpoint,
        patience=args.patience,
        num_fusion_layers=args.num_fusion_layers,
        position_embed_mode=args.position_embed_mode,
    )
