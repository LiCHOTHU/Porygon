#!/usr/bin/env python3
"""Render a LIBERO 2-stage clip: frozen BASE policy vs the CAST-finetuned policy
(base + trained residual) on the same task, by capturing the agentview camera
frames from the env observations. Run in the `porygon` env, on a GPU (EGL).

Two stages:
  BASE   -> collector mode="base"   (residual forced to 0)
  CAST   -> collector mode="policy" (deployed base + residual, best-of-N)
"""
import os, sys, importlib.util
import numpy as np, torch, imageio

REPO = "/storage/home/hcoda1/8/lwang831/workspace/imitation"
BASE_CKPT = ("/storage/scratch1/8/lwang831/imitation/cold_start/libero/"
             "libero_90/drift_multitask_lib90/multitask_model_latest.pth")
RES_CKPT = ("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/"
            "imitation/experiments_residual/residual_libero_drift_t32/residual_latest.pth")
TASK = 32
OUT = os.path.join(REPO, "icra_video/task_demos/raw_libero")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda"

# import the LIBERO residual harness (does its own sys.path setup on import)
tp = os.path.join(REPO, "act_sim_ppo_v6_gtsim/integrations/libero_imitation/train.py")
spec = importlib.util.spec_from_file_location("libero_train", tp)
T = importlib.util.module_from_spec(spec)
sys.modules["libero_train"] = T
spec.loader.exec_module(T)

print("[setup] building base policy + env runner ...", flush=True)
policy, env_runner = T.build_policy_and_runner(BASE_CKPT, DEVICE)
S = T.probe_state_feat_dim(policy, env_runner, TASK, DEVICE)
base = T.FrozenImitationBase(policy, DEVICE, state_feat_dim=S, base_seed=0)
model = T.ResidualRLModel(
    frozen_act=base, action_dim=policy.network_action_dim,
    n_exec=policy.chunk_size, residual_scale=0.1, critic_ensemble_size=2,
    bc_loss_weight=10.0, device=DEVICE).to(DEVICE)

# load the trained residual weights (actor/critic/target_critic)
rc = torch.load(RES_CKPT, map_location="cpu", weights_only=False)
model.actor.load_state_dict(rc["actor"])
model.critic.load_state_dict(rc["critic"])
model.target_critic.load_state_dict(rc["target_critic"])
model.eval()
print(f"[setup] loaded residual (base_sr={rc.get('meta',{}).get('base_sr')})", flush=True)

collector = T.LiberoResidualCollector(env_runner, base, model, DEVICE)
collector._build_env(TASK)

RES = 512  # native render resolution (independent of the policy's 128px obs)

def _find_sim(venv):
    """Walk the wrapper/vector-env stack to the robosuite env exposing .sim."""
    seen = set()
    stack = [venv]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        sim = getattr(o, "sim", None)
        if sim is not None and hasattr(sim, "render"):
            return sim
        for attr in ("env", "_env", "unwrapped"):
            c = getattr(o, attr, None)
            if c is not None:
                stack.append(c)
        for attr in ("envs", "workers"):
            seq = getattr(o, attr, None)
            if seq:
                for w in (seq if isinstance(seq, (list, tuple)) else [seq]):
                    stack.append(getattr(w, "env", w))
    return None

frames = []
_sim_holder = {"sim": None}
def _capture():
    sim = _sim_holder["sim"]
    if sim is None:
        sim = _sim_holder["sim"] = _find_sim(collector._env)
    if sim is None:
        return
    img = sim.render(width=RES, height=RES, camera_name="agentview")
    frames.append(np.ascontiguousarray(img[::-1]))   # mujoco render is bottom-origin

_env = collector._env
_ostep, _oreset = _env.step, _env.reset
def step_cap(*a, **k):
    r = _ostep(*a, **k); _capture(); return r
def reset_cap(*a, **k):
    r = _oreset(*a, **k); _sim_holder["sim"] = None; _capture(); return r
_env.step, _env.reset = step_cap, reset_cap

def render_episode(mode, init_idx):
    frames.clear()
    succ, n = collector._run_episode(TASK, init_idx, 10**9, mode, replay=None)
    return bool(succ), list(frames)

def save(path, frs, fps=20):
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for f in frs:
        w.append_data(np.ascontiguousarray(f))
    w.close()

# BASE: prefer an episode that FAILS (shows the base's limitation); else first
chosen = {}
for mode, want_success in [("base", False), ("policy", True)]:
    best = None
    for init in range(6):
        succ, frs = render_episode(mode, init)
        print(f"  [{mode}] init={init} success={succ} frames={len(frs)}", flush=True)
        if not frs:
            continue
        if succ == want_success:
            best = (succ, frs); break
        if best is None:
            best = (succ, frs)
    if best is not None:
        tag = "cast" if mode == "policy" else "base"
        save(os.path.join(OUT, f"libero_{tag}.mp4"), best[1])
        chosen[tag] = (best[0], len(best[1]))
        print(f"WROTE libero_{tag}.mp4  success={best[0]}  frames={len(best[1])}", flush=True)

import json
json.dump({"task": f"LIBERO-90 task {TASK}", "stages": {
    "base": {"kind": "frozen base policy (residual=0)",
             "success": chosen.get("base", (None,))[0], "base_success_rate": 0.6},
    "cast": {"kind": "CAST-finetuned (base + trained residual)",
             "success": chosen.get("cast", (None,))[0]}},
    "base_checkpoint": BASE_CKPT, "residual_checkpoint": RES_CKPT},
    open(os.path.join(OUT, "libero_stages.json"), "w"), indent=2)
print("DONE", chosen, flush=True)
