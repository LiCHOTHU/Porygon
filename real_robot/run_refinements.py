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
    bc_status = json.loads((bc / "status.json").read_text())
    if bc_status["state"] not in ("completed", "early_stopped"):
        raise RuntimeError("BC must finish before starting the RL stages")
    # Use the actually fine-tuned endpoint for the fixed-budget experiment,
    # not best.pt, which may intentionally retain the unmodified baseline.
    if not (bc / "final.pt").exists():
        shutil.copy2(bc / "latest.pt", bc / "final.pt")
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
        status({"state": "evaluating_offline", "stage": "action_diagnostics"}, output)
        subprocess.run([sys.executable, "-m", "real_robot.evaluate_comparison",
                        "--experiment-dir", str(output)], cwd=PROJECT, check=True)
        status({"state": "completed", "stages": ["bc_refinement", "dice_rl", "cast"]}, output)
    except BaseException as exc:
        status({"state": "failed", "error": str(exc)}, output)
        raise


if __name__ == "__main__":
    main()
