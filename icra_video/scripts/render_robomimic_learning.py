#!/usr/bin/env python3
"""Render a robomimic CAST policy rollout at a given training checkpoint, using
the harness's own env video recorder. One episode -> one mp4.

Run in the `dice-rl` conda env, on a GPU (robosuite offscreen render needs EGL).
Usage: python render_robomimic_learning.py <checkpoint.pth> <out.mp4> [n_steps]
"""
import os, sys, numpy as np, torch
from omegaconf import OmegaConf, open_dict

CKPT, OUT = sys.argv[1], sys.argv[2]
N_STEPS_OVERRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

# data dir env used by cfg interpolations (normalization_path = ${oc.env:DPPO_DATA_DIR}/...)
CED = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch"
os.environ.setdefault("DPPO_DATA_DIR", f"{CED}/dice_rl_official/data_dir")
os.environ.setdefault("DPPO_LOG_DIR", f"{CED}/dice_rl_official/log_dir")
os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
print("env_name:", cfg.get("env_name"), "step:", ck.get("step"), flush=True)

# force a single eval env that records video, minimal iterations
with open_dict(cfg):
    cfg.device = "cuda"
    render_dir = os.path.dirname(OUT)
    cfg.logdir = render_dir
    cfg.env.n_envs = 1
    if "eval_n_envs" in cfg:
        cfg.eval_n_envs = 1
    cfg.env.save_video = True
    if "train" in cfg and "render" in cfg.train:
        cfg.train.render.num = 1
        cfg.train.render.freq = 1
    # keep base_policy_path resolvable
print("building agent (env + model)...", flush=True)

# the agent __init__ reads HydraConfig.get().runtime.output_dir; we instantiate
# it directly (not under hydra.main), so set the Hydra singleton manually.
from hydra.core.hydra_config import HydraConfig
# set the singleton cfg directly (set_config() demands a structured HydraConf);
# get() returns instance.cfg.hydra, so we only need runtime.output_dir present.
HydraConfig.instance().cfg = OmegaConf.create(
    {"hydra": {"runtime": {"output_dir": render_dir}}})
with open_dict(cfg):
    cfg.wandb = None
    # This checkpoint predates several current train-config keys. They are all
    # optimizer/scheduler params, irrelevant to eval rendering; fill defaults so
    # the agent __init__ (which builds optimizers) doesn't crash on missing keys.
    tr = cfg.train
    def ensure(node, key, val):
        if key not in node:
            node[key] = val
    for sk in ["actor_lr_scheduler", "critic_lr_scheduler"]:
        if sk not in tr:
            tr[sk] = {}
        ensure(tr[sk], "first_cycle_steps", 1_000_000)
        ensure(tr[sk], "min_lr", 1e-6)
        ensure(tr[sk], "warmup_steps", 0)
    for k, v in {"actor_lr": 1e-4, "critic_lr": 1e-4, "actor_weight_decay": 0.0,
                 "critic_weight_decay": 0.0, "buffer_size": 10000, "eta": 1.0,
                 "gamma": 0.99, "replay_ratio": 1, "scale_reward_factor": 1.0,
                 "target_ema_rate": 0.005, "n_critic_warmup_itr": 0,
                 "batch_size": 256, "n_train_itr": 1, "save_model_freq": 10_000_000,
                 "val_freq": 10_000_000}.items():
        ensure(tr, k, v)
    if "render" not in tr:
        tr["render"] = {}
    ensure(tr["render"], "freq", 1)
    ensure(tr["render"], "num", 1)
    # single eval env, single episode, recorded
    cfg.eval_n_envs = 1
    cfg.env.n_envs = 1
    cfg.num_eval_episodes = 1
    # render the recorded video at 512x512 (state-based policy, so this only
    # affects the video resolution, never the policy input)
    def _bump_render_hw(node):
        try:
            items = node.items()
        except Exception:
            return
        for k, v in items:
            if k == "render_hw":
                node[k] = [512, 512]
            elif hasattr(v, "items"):
                _bump_render_hw(v)
    _bump_render_hw(cfg)

# correct agent for the CAST distill-residual model
from agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
agent = TrainDistillResidualFlowAgent(cfg)
agent.model.load_state_dict(ck["model_state_dict"])
agent.model.eval()

# force one recorded eval episode into render_dir
agent.render_video = True
agent.n_render = 1
agent.num_eval_episodes = 1
STEP = int(ck.get("step", 0))
print(f"running one recorded eval episode (step={STEP}) ...", flush=True)
agent.evaluate(total_steps=STEP)

# the env recorder wrote into agent.render_dir (= logdir/render) -> move to OUT
import glob, shutil
cands = sorted(glob.glob(os.path.join(agent.render_dir, f"{STEP}_eval_episode-*_env-0.mp4")))
if cands:
    shutil.move(cands[0], OUT)
    print(f"WROTE {OUT}", flush=True)
else:
    print("NO VIDEO PRODUCED; render_dir contents:",
          os.listdir(render_dir), flush=True)
