import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from real_robot.data import RealRobotDataset, load_stats
from real_robot.policy import RealRobotFlowPolicy


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for batch in loader:
            losses.append(model.compute_loss(move_batch(batch, device)).item())
    return float(np.mean(losses))


def save_checkpoint(path, model, optimizer, scheduler, scaler, cfg, epoch, step, val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "prompt": cfg["prompt"],
        "action_order": [f"left_joint_{i}" for i in range(1, 8)] + ["left_gripper"],
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="real_robot/config.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    train_set = RealRobotDataset(cfg["cache_dir"], "train", augment=True)
    val_set = RealRobotDataset(cfg["cache_dir"], "val", augment=False)
    loader_args = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": True,
        "persistent_workers": int(cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    stats = load_stats(cfg["cache_dir"])
    prompt_embedding = torch.from_numpy(
        np.load(Path(cfg["cache_dir"]) / "prompt_embedding.npy")
    )
    model = RealRobotFlowPolicy(cfg, stats, prompt_embedding).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]), betas=(0.95, 0.999),
    )
    total_steps = int(cfg["epochs"]) * len(train_loader)
    warmup = min(int(cfg["warmup_steps"]), max(1, total_steps // 10))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    start_epoch = 0
    step = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        step = int(checkpoint["step"])
        best_val = float(checkpoint["val_loss"])

    print(f"Training on {device}: {len(train_set)} train, {len(val_set)} val samples")
    history_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}",
                        disable=not sys.stderr.isatty())
        for batch_index, batch in enumerate(progress):
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.compute_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += loss.item()
            progress.set_postfix(loss=f"{running / (batch_index + 1):.4f}")

        train_loss = running / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1, "step": step, "train_loss": train_loss,
            "val_loss": val_loss, "lr": scheduler.get_last_lr()[0],
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        save_checkpoint(
            output / "latest.pt", model, optimizer, scheduler, scaler, cfg,
            epoch, step, val_loss,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output / "best.pt", model, optimizer, scheduler, scaler, cfg,
                epoch, step, val_loss,
            )
        print(json.dumps(record))


if __name__ == "__main__":
    main()
