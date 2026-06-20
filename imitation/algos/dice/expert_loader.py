"""RLPD expert-demo loader for DICE residual RL on LIBERO.

Fills a FROZEN ReplayBuffer with expert transitions built from the LIBERO BC
demos, encoded through the frozen BC policy so they live in the SAME space as
the online data the DiceCollector produces:

  - decision cadence: one transition every `action_horizon` env steps (the
    collector emits a chunk decision each time its action queue drains)
  - cond            : bc_policy.get_cond(preprocess_input(batch)) -- the same
                      frozen-encoder embedding the student/critic consume
  - action          : the dataset's chunk_size-step action window, normalized
                      (same space as the collector's a_norm)
  - reward/done     : 1.0 / True on the LAST decision of each demo. LIBERO
                      demos all terminate in success, which mirrors the
                      official repo's ph_finetune dataset ("trajectories
                      truncated to have exactly one success at the end to
                      ensure the value learning between offline data and
                      online data is consistent")
  - n-step + mc_return: computed by ReplayBuffer.add_trajectory, identical to
                      the online path

The returned buffer is intended to be attached as the online ReplayBuffer's
`expert_dataset`; ReplayBuffer.sample then mixes expert_ratio * B expert rows
into every minibatch (RLPD symmetric sampling).
"""

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import default_collate

from imitation.algos.dice.replay_buffer import ReplayBuffer
from imitation.utils.utils import map_tensor_to_device


@torch.no_grad()
def build_expert_buffer(cfg, bc_policy, task_indices, cond_shape, device,
                        encode_batch_size: int = 32, logger=None):
    """Build a frozen expert ReplayBuffer from the LIBERO demos of `task_indices`.

    Encodes demo observations through the frozen BC encoder in batches, then
    pushes one trajectory per demo via add_trajectory(data_source=1).
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)

    task_indices = [int(t) for t in task_indices]
    action_horizon = int(bc_policy.action_horizon)

    # The BC dataset config already encodes seq_len=chunk_size, frame_stack and
    # the obs modalities the encoder was trained on; we only force task_subset
    # to this run's tasks and load_obs=True (RL configs ship load_obs=False).
    _log(f"RLPD: building expert dataset for tasks {task_indices} "
         f"(n_demos={cfg.task.demos_per_env}/task, stride={action_horizon})...")
    dataset = instantiate(cfg.task.dataset, task_subset=task_indices, load_obs=True)
    assert isinstance(dataset, ConcatDataset), type(dataset)
    assert len(dataset.datasets) == len(task_indices)

    # Count decisions to size the buffer exactly.
    per_task_meta = []
    total_decisions = 0
    for j, svl in enumerate(dataset.datasets):
        sd = svl.sequence_dataset
        demos = []
        for demo_id in sd.demos:
            start = sd._demo_id_to_start_indices[demo_id]
            length = sd._demo_id_to_demo_length[demo_id]
            t_steps = list(range(0, length, action_horizon))
            demos.append((start, t_steps))
            total_decisions += len(t_steps)
        per_task_meta.append(demos)

    expert_buf = ReplayBuffer(
        max_size=total_decisions,
        cond_shape=tuple(cond_shape),
        horizon=bc_policy.chunk_size,
        action_dim=bc_policy.network_action_dim,
        device=device,
        gamma=cfg.dice.gamma,
        use_n_step=cfg.dice.get("use_n_step", False),
        n_step=cfg.dice.get("n_step", 1),
    )

    # Encode demo decisions through the frozen BC encoder, batched. Disable
    # augmentation: online cond comes from clean env frames, expert cond must
    # match that distribution.
    was_training = bc_policy.training
    saved_aug = getattr(bc_policy, "use_augmentation", False)
    bc_policy.eval()
    bc_policy.use_augmentation = False
    try:
        n_demos_done = 0
        for j, svl in enumerate(dataset.datasets):
            true_task_id = task_indices[j]
            for (start, t_steps) in per_task_meta[j]:
                conds, actions = [], []
                for b0 in range(0, len(t_steps), encode_batch_size):
                    idxs = t_steps[b0:b0 + encode_batch_size]
                    items = [svl[start + t] for t in idxs]
                    batch = default_collate(items)
                    # Dataset task_id is the subset-relative enumeration index;
                    # the encoder (if it consumes task_id at all) was trained
                    # with GLOBAL benchmark indices via _make_batch. Override.
                    batch["task_id"] = torch.full((len(items),), true_task_id, dtype=torch.long)
                    batch = map_tensor_to_device(batch, device)
                    batch = bc_policy.preprocess_input(batch, train_mode=True)
                    conds.append(bc_policy.get_cond(batch).cpu())
                    actions.append(batch["actions"].to(torch.float32).cpu())
                cond_t = torch.cat(conds, dim=0)        # (T, num_enc, hidden)
                act_t = torch.cat(actions, dim=0)       # (T, chunk, A) normalized
                T = cond_t.shape[0]
                traj = []
                for t in range(T):
                    last = (t == T - 1)
                    traj.append({
                        "cond": cond_t[t],
                        "noise": torch.randn(bc_policy.chunk_size, bc_policy.network_action_dim),
                        "action": act_t[t],
                        "reward": 1.0 if last else 0.0,
                        "done": bool(last),
                    })
                expert_buf.add_trajectory(traj, data_source=1)
                n_demos_done += 1
            _log(f"RLPD: task {true_task_id} encoded "
                 f"({len(per_task_meta[j])} demos, buffer={len(expert_buf)})")
    finally:
        bc_policy.use_augmentation = saved_aug
        if was_training:
            bc_policy.train()

    assert len(expert_buf) == total_decisions, (len(expert_buf), total_decisions)
    _log(f"RLPD: expert buffer ready -- {n_demos_done} demos, "
         f"{total_decisions} transitions, data_source=1")
    return expert_buf
