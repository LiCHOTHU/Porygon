"""Background-capable fine-tuning with fixed holdouts and original-model baseline."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from real_robot.data import RealRobotDataset, load_stats
from real_robot.policy import RealRobotFlowPolicy
from real_robot.train import move_batch


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def update_plot(output, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in (("train_loss", "Training"), ("val_loss", "Validation")):
        axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
    axes[0].set_title("Flow loss (expanded training normalization)")
    axes[0].set_ylabel("Velocity MSE")
    measured = [r for r in rows if "action_mse" in r["validation"]]
    axes[1].plot([r["epoch"] for r in measured],
                 [r["validation"]["action_mse"] for r in measured], "o-", label="Continued policy")
    axes[1].axhline(baseline["action_mse"], color="gray", linestyle="--", label="Original best checkpoint")
    axes[1].set_title("Held-out action error (same scale / fixed noise)")
    axes[1].set_ylabel("Normalized action MSE — lower is better")
    for ax in axes:
        ax.set_xlabel("Additional training epoch")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Jigglypuff — original + supplementary demonstrations")
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160)
    plt.close(fig)


def evaluate(model, loaders, stats, seed, actions=False):
    model.eval()
    result = {}
    total_loss = total_count = 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
        for group_index, (name, loader) in enumerate(loaders.items()):
            torch.manual_seed(seed + group_index)
            loss_sum = count = 0
            with torch.autocast("cuda", dtype=torch.float16):
                for batch in loader:
                    batch = move_batch(batch, "cuda")
                    n = len(batch["actions"])
                    loss_sum += model.compute_loss(batch).item() * n
                    count += n
            result[name] = {"flow_loss": loss_sum / count, "samples": count}
            total_loss += loss_sum
            total_count += count
        if actions:
            # Always use the same physical-action scaling for the original model
            # and fine-tuned models, even though their internal normalizers differ.
            scale = (stats["action_max"] - stats["action_min"]).cuda().clamp_min(1e-6)
            score_sum = 0
            for group_index, (name, loader) in enumerate(loaders.items()):
                torch.manual_seed(seed + 10000 + group_index)
                mse = hold_mse = count = 0
                mae = torch.zeros(8, device="cuda")
                first = torch.zeros_like(mae)
                with torch.autocast("cuda", dtype=torch.float16):
                    for batch in loader:
                        batch = move_batch(batch, "cuda")
                        prediction = model.sample_actions(batch)
                        if not torch.isfinite(prediction).all():
                            raise RuntimeError("Non-finite sampled actions")
                        error = prediction - batch["actions"]
                        hold_error = batch["proprio"][:, None] - batch["actions"]
                        n = len(prediction)
                        mse += ((2 * error / scale)**2).mean().item() * n
                        hold_mse += ((2 * hold_error / scale)**2).mean().item() * n
                        mae += error.abs().mean(1).sum(0)
                        first += error[:, 0].abs().sum(0)
                        count += n
                result[name].update(action_mse=mse / count, hold_action_mse=hold_mse / count,
                                    chunk_mae=(mae / count).cpu().tolist(),
                                    first_step_mae=(first / count).cpu().tolist())
                score_sum += mse
            result["action_mse"] = score_sum / total_count
    result["val_loss"] = total_loss / total_count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="real_robot/config_jigglypuff_continued.yaml")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.background:
        if (output / "metrics.jsonl").exists():
            raise RuntimeError("Run already exists; choose a new output_dir")
        with (output / "train.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "real_robot.continue_train", "--config",
                 str(Path(args.config).resolve())], cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        (output / "train.pid").write_text(str(process.pid) + "\n")
        print(f"Background training PID {process.pid}; log: {output / 'train.log'}")
        return
    run_lock = (output / "training.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "metrics.jsonl").exists():
        raise RuntimeError("Run already exists; choose a new output_dir rather than overwrite it")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    cfg["selection_metric"] = "fixed_seed_validation_normalized_action_mse"
    cfg["normalization"] = "combined_training_data; weights initialized from original best"
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = json.loads((Path(cfg["cache_dir"]) / "manifest.json").read_text())
    write_json(output / "data_manifest.json", manifest)
    train = RealRobotDataset(cfg["cache_dir"], "train", augment=True)
    val = RealRobotDataset(cfg["cache_dir"], "val", augment=False)
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)
    sources = {ep["name"]: Path(ep["source_root"]).name for ep in manifest["episodes"]}
    groups = {}
    for index, (episode, _) in enumerate(val.samples):
        groups.setdefault(sources[episode], []).append(index)
    val_loaders = {name: DataLoader(Subset(val, indices), batch_size=32, num_workers=0)
                   for name, indices in sorted(groups.items())}
    source = torch.load(cfg["initial_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("action_dim", "proprio_dim", "chunk_size", "image_size", "hidden_dim",
                "num_blocks", "num_heads", "feedforward_dim", "fps", "prompt"):
        if cfg[key] != source["config"][key]:
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    stats = load_stats(cfg["cache_dir"])
    model = RealRobotFlowPolicy(cfg, stats, source["model"]["prompt_embedding"],
                                pretrained_backbone=False).cuda()
    model.load_state_dict(source["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  betas=(.95, .999), weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(loader)
    warmup = min(cfg["warmup_steps"], max(1, total_steps // 10))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / (total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda")
    best_score = float("inf")
    best_epoch = 0
    step = 0

    def save(name, epoch, validation):
        temp = output / (name + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg, "epoch": epoch - 1, "step": step,
                    "val_loss": validation["val_loss"], "validation": validation,
                    "best_action_mse": best_score, "best_additional_epoch": best_epoch,
                    "prompt": cfg["prompt"], "action_order": source["action_order"],
                    "parent_checkpoint": cfg["initial_checkpoint"],
                    "parent_epoch": source["epoch"] + 1}, temp)
        temp.replace(output / name)

    status = {"state": "baseline_evaluation", "train_samples": len(train), "val_samples": len(val),
              "max_additional_epochs": cfg["epochs"], "parent_epoch": source["epoch"] + 1}
    write_json(output / "status.json", status)
    print(f"Original epoch {source['epoch'] + 1}; {len(train)} train / {len(val)} validation samples", flush=True)
    baseline = evaluate(model, val_loaders, stats, cfg["validation_seed"], actions=True)
    write_json(output / "baseline.json", baseline)
    best_score = baseline["action_mse"]
    save("best.pt", 0, baseline)  # Original model remains incumbent until beaten.
    print("Original checkpoint baseline: " + json.dumps(baseline), flush=True)
    old_stats = {key: model.state_dict()[key].cpu().tolist() for key in stats}
    for key, value in stats.items():
        getattr(model, key).copy_(value.cuda())
    write_json(output / "normalization.json", {"original": old_stats,
                                               "continued": {k: v.tolist() for k, v in stats.items()}})
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            started = time.monotonic()
            model.train()
            total = count = 0
            for batch in loader:
                batch = move_batch(batch, "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                step += 1
                total += loss.item() * len(batch["actions"])
                count += len(batch["actions"])
            action_eval = epoch == 1 or epoch % cfg["action_eval_every"] == 0
            validation = evaluate(model, val_loaders, stats, cfg["validation_seed"], actions=action_eval)
            improved = action_eval and validation["action_mse"] < best_score
            if improved:
                best_score, best_epoch = validation["action_mse"], epoch
            record = {"epoch": epoch, "step": step, "train_loss": total / count,
                      "val_loss": validation["val_loss"], "validation": validation,
                      "best_action_mse": best_score, "best_epoch": best_epoch,
                      "lr": scheduler.get_last_lr()[0], "seconds": time.monotonic() - started}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            save("latest.pt", epoch, validation)
            if improved:
                save("best.pt", epoch, validation)
            status.update(state="running", additional_epochs_completed=epoch, best_epoch=best_epoch,
                          best_action_mse=best_score, baseline_action_mse=baseline["action_mse"],
                          latest_train_loss=record["train_loss"], latest_val_loss=record["val_loss"])
            write_json(output / "status.json", status)
            if action_eval:
                update_plot(output, baseline)
            print(json.dumps(record), flush=True)
            if epoch - best_epoch >= cfg["patience"]:
                status["state"] = "early_stopped"
                break
        else:
            status["state"] = "completed"
    except BaseException as exc:
        status.update(state="interrupted_or_failed", error=str(exc))
        write_json(output / "status.json", status)
        raise
    write_json(output / "status.json", status)
    print("Finished: " + json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
