"""Offline prefix-action diagnostics for all four conditions; no robot rollout."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from real_robot.prepare_comparison import OUTPUT
from real_robot.data import RealRobotDataset
from real_robot.infer import JigglypuffPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=OUTPUT)
    directory = parser.parse_args().experiment_dir
    experiment = json.loads((directory / "experiment.json").read_text())
    source = torch.load(experiment["base_checkpoint"], map_location="cpu", weights_only=False)
    dataset = RealRobotDataset(source["config"]["cache_dir"], "val")
    scale = (source["model"]["action_max"] - source["model"]["action_min"]).cuda().clamp_min(1e-6)
    paths = {"base_fm": experiment["base_checkpoint"],
             **{mode: str(directory / mode / "final.pt") for mode in ("bc_refinement", "dice_rl", "cast")}}
    report = {"note": "Offline held-out eight-step action errors; NOT real-robot task success rates", "methods": {}}
    for name, path in paths.items():
        policy = JigglypuffPolicy(path)
        torch.manual_seed(1729)
        mae = torch.zeros(8, device="cuda")
        first = torch.zeros_like(mae)
        total = count = 0
        for batch in DataLoader(dataset, batch_size=32, num_workers=2):
            batch = {k: v.cuda() for k, v in batch.items()}
            predicted = policy.model.sample_actions(batch)[:, :8]
            assert predicted.shape[1:] == (8, 8) and torch.isfinite(predicted).all()
            error = predicted - batch["actions"][:, :8]
            mae += error.abs().mean(1).sum(0)
            first += error[:, 0].abs().sum(0)
            total += ((2 * error / scale)**2).mean().item() * len(predicted)
            count += len(predicted)
        report["methods"][name] = {"checkpoint": path, "samples": count,
                                    "prefix_normalized_mse": total / count,
                                    "prefix_mae": (mae / count).cpu().tolist(),
                                    "first_step_mae": (first / count).cpu().tolist()}
        del policy
    (directory / "offline_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
