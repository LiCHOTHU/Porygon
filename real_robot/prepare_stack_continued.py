"""Freeze original and supplementary stacking splits for checkpoint refinement."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from real_robot.prepare_data import read_joint_csv


def main():
    project = Path(__file__).resolve().parents[1]
    workspace = Path("/home/pair/Desktop/openarm_exp/openarm_ws")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=project / "real_robot/runs/stack/best.pt")
    parser.add_argument("--new-data-root", type=Path, default=workspace / "data_collection_stack_supplementsary")
    parser.add_argument("--run-name", default="stack_continued")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    args = parser.parse_args()
    if not args.run_name.replace("_", "").isalnum():
        parser.error("run-name must contain only letters, digits, and underscores")
    supplementary = args.new_data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original = Path(source["config"]["data_root"])
    roots = [Path(p) for p in source["config"].get("data_roots", [str(original)])]
    if supplementary in roots:
        parser.error("new-data-root is already in the parent training data")
    roots.append(supplementary)
    run = project / "real_robot/runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    config_path = project / "real_robot" / f"config_{args.run_name}.yaml"
    if config_path.exists() or (run / "metrics.jsonl").exists():
        raise RuntimeError("Continuation config/run already exists; do not resample or overwrite it")
    old_manifest = json.loads((Path(source["config"]["cache_dir"]) / "manifest.json").read_text())
    original_val = {e["name"] for e in old_manifest["episodes"] if e["split"] == "val"}
    usable, excluded = [], []
    for root in roots:
        for take in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            try:
                jt, _ = read_joint_csv(take / "leader_joint.csv")
                meta = json.loads((take / "take_meta.json").read_text())
                if meta["fps"] != 15:
                    raise ValueError("Unexpected recording FPS")
                end = float(meta["ended_at_epoch_s"])
                starts = [jt[0]]
                for pattern in ("chest_rgb_*.avi", "left_wrist_rgb_*.avi"):
                    files = list(take.glob(pattern))
                    if len(files) != 1:
                        raise ValueError(f"Expected one {pattern}")
                    cap = cv2.VideoCapture(str(files[0]))
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if count <= 30:
                        raise ValueError("Video too short for 30-step action chunks")
                    starts.append(end - (count - 1) / 15)
                if max(starts) + 2 >= jt[-1]:
                    raise ValueError("Insufficient synchronized recording duration")
                usable.append(take)
            except Exception as exc:
                excluded.append({"name": take.name, "source_root": str(root), "reason": str(exc)})
    supplementary_takes = [p for p in usable if p.parent == supplementary]
    if len(supplementary_takes) < 2:
        raise RuntimeError("Need at least two usable supplementary takes")
    indices = np.random.default_rng(42).choice(len(supplementary_takes),
                                              max(1, round(.2 * len(supplementary_takes))), replace=False)
    new_val = {supplementary_takes[i].name for i in indices}
    if not original_val <= {p.name for p in usable}:
        raise RuntimeError("An original validation take is unavailable")
    cfg = dict(source["config"])
    cfg.update(data_roots=[str(p) for p in roots],
               include_takes=[p.name for p in usable], validation_takes=sorted(original_val | new_val),
               preparation_exclusions=excluded,
               cache_dir=str(project / "real_robot/cache" / args.run_name),
               output_dir=str(run), initial_checkpoint=str(checkpoint),
               export_dir=str(workspace / "policy_checkpoints" / args.run_name),
               plot_title=f"Stacking — {args.run_name}",
               freeze_normalizer=False, epochs=200, patience=40, learning_rate=args.learning_rate,
               validation_seed=1729, action_eval_every=5, selection_horizon=8)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    report = {"usable_original": sum(p.parent == original for p in usable),
              "usable_supplementary": len(supplementary_takes),
              "usable_by_root": {str(root): sum(p.parent == root for p in usable) for root in roots},
              "train_takes": len(usable) - len(original_val | new_val),
              "validation_takes": len(original_val | new_val),
              "excluded": excluded, "initial_checkpoint": str(checkpoint),
              "initial_epoch": source["epoch"] + 1}
    (run / "data_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
