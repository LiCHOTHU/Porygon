#!/usr/bin/env python3
"""Render DMC task clips by rolling out the trained SAC teacher policy in the
actual dm_control simulator -- the same policy that seeds each CAST base, so
these show competent behaviour on the exact tasks in the DMC results table.

Real policy rollouts (not scripted). Run in the `dice-rl` env (has dm_control).
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "icra_video" / "task_demos"
SAC = Path("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dmc_base")

H, LOG_STD_MIN, LOG_STD_MAX = 256, -5.0, 2.0


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, o))


class Actor(nn.Module):
    def __init__(self, odim, adim):
        super().__init__(); self.net = mlp(odim, 2 * adim); self.adim = adim

    def det(self, s):
        mu, _ = self.net(s).chunk(2, -1)
        return torch.tanh(mu)


class Writer:
    def __init__(self, path, fps):
        import imageio
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                    quality=8, macro_block_size=8)
        self.count = 0

    def write(self, f):
        self.w.append_data(np.ascontiguousarray(f)); self.count += 1

    def close(self):
        self.w.close()


def _rollout(actor, domain, task, seconds, seed, low, high, flatten, suite):
    """Roll out one actor, return (frames_stride_writer_fn ready list, total_reward)."""
    env = suite.load(domain, task, task_kwargs={"random": seed})
    dt = env.control_timestep()
    ts = env.reset(); so = flatten(ts.observation)
    n_steps = int(round(seconds / dt))
    fps = 30
    stride = max(1, int(round(1.0 / (fps * dt))))
    frames = []; total_r = 0.0
    for step in range(n_steps):
        with torch.no_grad():
            a = actor.det(torch.as_tensor(so)[None])[0].numpy()
        ts = env.step(low + (a + 1.0) * 0.5 * (high - low))
        total_r += float(ts.reward or 0.0)
        if step % stride == 0:
            frames.append(env.physics.render(height=480, width=640, camera_id=0))
        so = flatten(ts.observation)
        if ts.last():
            ts = env.reset(); so = flatten(ts.observation)
    return frames, total_r, fps


def _banner(frame, big, small):
    """Two-line caption strip: stage name (bold-ish) + the measured return."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 52], fill=(0, 0, 0))
    try:
        fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        fs = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        fb = fs = None
    d.text((12, 4), big, fill=(255, 255, 255), font=fb)
    d.text((12, 32), small, fill=(180, 220, 255), font=fs)
    return np.asarray(im)


def render_learning(domain, task, ckpt, stages_spec, seconds=6.0, seed=7):
    """One clip showing the policy gradually learning: roll out three chosen
    training stages (early -> mid -> final/high-success) on the same task, each
    with a two-line caption, concatenated into a single before-to-after video.
    `stages_spec` = list of (snapshot_index_or_'final', STAGE_LABEL)."""
    from dm_control import suite
    from env.gym_utils.wrapper.dmc_lowdim import flatten_dmc_obs
    env = suite.load(domain, task, task_kwargs={"random": seed})
    spec = env.action_spec(); adim = int(np.prod(spec.shape))
    low = np.asarray(spec.minimum, np.float32).reshape(-1)
    high = np.asarray(spec.maximum, np.float32).reshape(-1)
    odim = flatten_dmc_obs(env.reset().observation).shape[0]
    sd = torch.load(ckpt, map_location="cpu")
    actor = Actor(odim, adim); actor.eval()
    label = f"dmc_{domain}_{task}_learning"
    writer = Writer(OUT / f"{label}.mp4", 30)
    snaps = list(sd["ckpts"]) + [sd["actor"]]
    stages = []
    for which, stage_label in stages_spec:
        state = snaps[-1] if which == "final" else snaps[which]
        actor.load_state_dict(state)
        frames, r, _ = _rollout(actor, domain, task, seconds, seed, low, high,
                                flatten_dmc_obs, suite)
        arrow = "" if not stages else ("  ▲ improving" if r > stages[-1]["return_over_clip"] else "")
        for f in frames:
            writer.write(_banner(f, f"{stage_label}",
                                 f"{domain}-{task}   episode return {r:.0f}{arrow}"))
        stages.append({"stage": stage_label, "snapshot": which,
                       "return_over_clip": round(r, 1)})
        print(f"  {label} [{stage_label}]: return {r:.1f}", flush=True)
    writer.close()
    (OUT / f"{label}.json").write_text(json.dumps(
        {"clip": label, "benchmark": "DMC", "domain": domain, "task": task,
         "kind": "learning progress: three training stages, early -> mid -> final",
         "policy_checkpoint": str(ckpt), "stages": stages, "fps": 30,
         "seconds_per_stage": seconds, "resolution": [640, 480]}, indent=2) + "\n")
    print("WROTE", label, flush=True)


def render_task(domain, task, ckpt, seconds=9.0, seed=1234):
    from dm_control import suite
    from env.gym_utils.wrapper.dmc_lowdim import flatten_dmc_obs
    env = suite.load(domain, task, task_kwargs={"random": seed})
    spec = env.action_spec()
    adim = int(np.prod(spec.shape))
    low = np.asarray(spec.minimum, np.float32).reshape(-1)
    high = np.asarray(spec.maximum, np.float32).reshape(-1)
    ts = env.reset()
    odim = flatten_dmc_obs(ts.observation).shape[0]
    actor = Actor(odim, adim)
    sd = torch.load(ckpt, map_location="cpu")
    actor.load_state_dict(sd["actor"]); actor.eval()

    dt = env.control_timestep()
    fps = 30
    n_steps = int(round(seconds / dt))
    stride = max(1, int(round(1.0 / (fps * dt))))
    label = f"dmc_{domain}_{task}"
    writer = Writer(OUT / f"{label}.mp4", fps)
    so = flatten_dmc_obs(ts.observation); total_r = 0.0
    for step in range(n_steps):
        with torch.no_grad():
            a = actor.det(torch.as_tensor(so)[None])[0].numpy()
        env_a = low + (a + 1.0) * 0.5 * (high - low)   # [-1,1] -> action range
        ts = env.step(env_a)
        total_r += float(ts.reward or 0.0)
        if step % stride == 0:
            writer.write(env.physics.render(height=480, width=640, camera_id=0))
        so = flatten_dmc_obs(ts.observation)
        if ts.last():
            ts = env.reset(); so = flatten_dmc_obs(ts.observation)
    writer.close()
    rec = {"clip": label, "benchmark": "DMC", "domain": domain, "task": task,
           "kind": "trained SAC teacher policy rollout in dm_control",
           "cast_evaluation": False, "policy_checkpoint": str(ckpt),
           "frames": writer.count, "fps": fps, "seconds": seconds,
           "episode_return_over_clip": round(total_r, 1),
           "resolution": [640, 480], "seed": seed}
    (OUT / f"{label}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("WROTE", label, writer.count, "frames  return", round(total_r, 1), flush=True)


# lean set: two competent task clips + one gradual-learning clip
TASKS = [
    ("cartpole", "swingup", "sac_cartpole_swingup.pt"),
    ("finger", "spin", "sac_finger_spin.pt"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for domain, task, ck in TASKS:
        try:
            render_task(domain, task, SAC / ck)
        except Exception as e:
            print("FAILED", domain, task, repr(e), flush=True)
    # the "how our policy gradually learned" clip: finger-spin has the cleanest
    # early -> mid -> final(high-success) progression (returns 656/863/919).
    try:
        render_learning("walker", "run", SAC / "sac_walker_run.pt",
                        stages_spec=[(0, "EARLY  (start of training)"),
                                     (2, "MID-TRAINING"),
                                     ("final", "FINAL  (converged)")],
                        seconds=6.0)
    except Exception as e:
        print("FAILED learning", repr(e), flush=True)
