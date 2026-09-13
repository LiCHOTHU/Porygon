"""Evaluate predicted joint chunks against held-out demonstrations, without ROS."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from real_robot.data import RealRobotDataset
from real_robot.infer import JigglypuffPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    torch.manual_seed(0)
    policy = JigglypuffPolicy(args.checkpoint)
    dataset = RealRobotDataset(cfg["cache_dir"], "val")
    sums = {key: torch.zeros(8, device="cuda") for key in
            ("first_step_mae", "chunk_mae", "hold_first_step_mae", "hold_chunk_mae")}
    count = 0
    for batch in DataLoader(dataset, batch_size=32, num_workers=2):
        batch = {key: value.cuda() for key, value in batch.items()}
        predicted = policy.model.sample_actions(batch)
        if not torch.isfinite(predicted).all():
            raise RuntimeError("Non-finite sampled action")
        error = (predicted - batch["actions"]).abs()
        hold_error = (batch["proprio"][:, None] - batch["actions"]).abs()
        sums["first_step_mae"] += error[:, 0].sum(0)
        sums["chunk_mae"] += error.mean(1).sum(0)
        sums["hold_first_step_mae"] += hold_error[:, 0].sum(0)
        sums["hold_chunk_mae"] += hold_error.mean(1).sum(0)
        count += len(predicted)
    result = {key: (value / count).cpu().tolist() for key, value in sums.items()}
    result.update(checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint["epoch"] + 1,
                  validation_samples=count, prompt=checkpoint["prompt"],
                  action_order=checkpoint["action_order"],
                  note="Offline single-seed sample errors; no robot task-success measurement.")
    print(json.dumps(result, indent=2))
    output = args.output or args.checkpoint.parent / "offline_evaluation.json"
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
