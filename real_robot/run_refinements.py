"""Supervise sequential offline refinements; stop the queue if a stage fails."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from real_robot.prepare_comparison import OUTPUT, PROJECT


def export_model(source, destination):
    import torch
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    for key in ("optimizer", "optimizers", "scheduler", "schedulers", "scaler",
                "actor_optimizer", "critic_optimizer", "critic_state", "target_critic_state"):
        checkpoint.pop(key, None)
    checkpoint["deployment_source"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(destination)


def status(value, directory=OUTPUT):
    path = directory / "pipeline_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--experiment-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.experiment_dir.resolve()
    if args.background:
        with (output / "pipeline.log").open("a") as log:
            p = subprocess.Popen([sys.executable, "-u", "-m", "real_robot.run_refinements",
                                  "--experiment-dir", str(output)],
                                 cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True)
        (output / "pipeline.pid").write_text(str(p.pid) + "\n")
        print(f"Sequential runner PID: {p.pid}")
        return
    lock = (output / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    experiment = json.loads((output / "experiment.json").read_text())
    base = Path(experiment["base_checkpoint"])
    if hashlib.sha256(base.read_bytes()).hexdigest() != experiment["base_sha256"]:
        raise RuntimeError("Base checkpoint changed after dataset manifest was frozen")
    bc = output / "bc_refinement"
    export = Path(experiment["export_dir"]) if experiment.get("export_dir") else None
    if export:
        export_model(base, export / "base.pt")
    if not (bc / "status.json").exists():
        bc.mkdir(exist_ok=True)
        with (bc / "train.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "real_robot.continue_train",
                                        "--config", experiment["bc_config"]],
                                       cwd=PROJECT, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
            status({"state": "running", "stage": "bc_refinement", "child_pid": process.pid,
                    "order": ["bc_refinement", "dice_rl", "cast"]}, output)
            if process.wait() != 0:
                status({"state": "failed", "stage": "bc_refinement"}, output)
                raise RuntimeError("BC refinement failed; inspect its train.log")
    bc_status = json.loads((bc / "status.json").read_text())
    if bc_status["state"] not in ("completed", "early_stopped"):
        raise RuntimeError("BC must finish before starting the RL stages")
    # Use the actually fine-tuned endpoint for the fixed-budget experiment,
    # not best.pt, which may intentionally retain the unmodified baseline.
    if not (bc / "final.pt").exists():
        shutil.copy2(bc / "latest.pt", bc / "final.pt")
    if export:
        export_model(bc / "final.pt", export / "bc_refinement/final.pt")
    try:
        for mode in ("dice_rl", "cast"):
            directory = output / mode
            directory.mkdir(exist_ok=True)
            if (directory / "status.json").exists():
                previous = json.loads((directory / "status.json").read_text())
                if previous["state"] == "completed":
                    continue
                raise RuntimeError(f"Existing unfinished {mode} run requires inspection")
            with (directory / "train.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-u", "-m", "real_robot.offline_rl", "--mode", mode,
                                            "--experiment-dir", str(output)],
                                           cwd=PROJECT, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
                status({"state": "running", "stage": mode, "child_pid": process.pid,
                        "order": ["bc_refinement", "dice_rl", "cast"]}, output)
                print(f"Starting {mode}, PID {process.pid}", flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f"{mode} failed; inspect {directory / 'train.log'}")
            if export:
                export_model(directory / "final.pt", export / mode / "final.pt")
        status({"state": "evaluating_offline", "stage": "action_diagnostics"}, output)
        subprocess.run([sys.executable, "-m", "real_robot.evaluate_comparison",
                        "--experiment-dir", str(output)], cwd=PROJECT, check=True)
        status({"state": "completed", "stages": ["bc_refinement", "dice_rl", "cast"]}, output)
    except BaseException as exc:
        status({"state": "failed", "error": str(exc)}, output)
        raise


if __name__ == "__main__":
    main()
