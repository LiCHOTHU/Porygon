"""Verify experiment provenance and the completed BC endpoint normalization."""
import json
import hashlib
import torch

from real_robot.prepare_comparison import OUTPUT


def main():
    experiment = json.loads((OUTPUT / "experiment.json").read_text())
    base = torch.load(experiment["base_checkpoint"], map_location="cpu", weights_only=False)
    bc = torch.load(OUTPUT / "bc_refinement/final.pt", map_location="cpu", weights_only=False)
    keys = ("action_min", "action_max", "proprio_min", "proprio_max", "prompt_embedding")
    assert all(torch.equal(base["model"][k], bc["model"][k]) for k in keys)
    assert len(experiment["expert_demonstrations"]) == 30
    replay = torch.load(OUTPUT / "replay.pt", weights_only=False)
    assert int(replay["done"].sum()) == 48 and int(replay["reward"].sum()) == 41
    report = {"base_retrained": False, "bc_additional_epochs": bc["epoch"] + 1,
              "normalizers_and_prompt_preserved": True,
              "expert_demos": 30, "reviewed_rollouts": len(experiment["policy_rollouts"]),
              "reviewed_successes": sum(r["success"] for r in experiment["policy_rollouts"]),
              "reviewed_failures": sum(not r["success"] for r in experiment["policy_rollouts"]),
              "expert_success_labels": "Assumed successful per the user-designated supplementary demo set",
              "replay_sha256": hashlib.sha256((OUTPUT / "replay.pt").read_bytes()).hexdigest()}
    report["bc_weights_changed"] = any(not torch.equal(value, bc["model"][key])
                                       for key, value in base["model"].items() if key.endswith("weight"))
    assert report["bc_weights_changed"]
    for mode in ("dice_rl", "cast"):
        path = OUTPUT / mode / "final.pt"
        if path.exists():
            final = torch.load(path, map_location="cpu", weights_only=False)
            assert all(torch.equal(value, final["model"][key]) for key, value in base["model"].items())
            report[mode] = {"updates": final["rl_update"], "frozen_base_exactly_preserved": True}
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2))
    # Fix metadata from the earlier generic fine-tuner: report the actual frozen
    # normalization, with raw dataset statistics retained under a separate key.
    path = OUTPUT / "bc_refinement/normalization.json"
    stats = json.loads(path.read_text())
    if "training_data" not in stats:
        stats["training_data"] = stats["continued"]
    stats["continued"] = {k: bc["model"][k].tolist() for k in keys if k != "prompt_embedding"}
    path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
