"""Direct measurement for the paper's core premise: a critic trained on
narrow demo coverage gives unreliable advice off-manifold. For a REAL
trained critic (from a finished square run), walk from demo actions along
(a) the critic's own gradient direction and (b) random directions, and
record Q(s, a + r*d) as a function of distance r. If Q keeps climbing along
its own gradient with no maximum near the data, the critic literally
believes that leaving the manifold is always better -- the 'compass with no
sense of distance' the method section claims."""
import sys, os, glob, numpy as np, torch, hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

RUN = sys.argv[1]            # run dir containing .hydra + checkpoint/
OUT = sys.argv[2]
cands = glob.glob(f"{RUN}/model_step_*.pth") + glob.glob(f"{RUN}/checkpoint/model_step_*.pth") \
        + glob.glob(f"{RUN}/checkpoint/state_*.pt")
assert cands, f"no checkpoints under {RUN}"
def _step(p):
    import re
    m = re.search(r"(\d+)\.(pth|pt)$", p)
    return int(m.group(1)) if m else -1
ck = sorted(cands, key=_step)[-1]
cfg = OmegaConf.load(f"{RUN}/.hydra/config.yaml")
model = hydra.utils.instantiate(cfg.model)
sd = torch.load(ck, map_location="cuda", weights_only=False)
sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
missing = model.load_state_dict(sd, strict=False)
print("load: missing", len(missing.missing_keys), "unexpected", len(missing.unexpected_keys))
model.cuda().eval()
print("loaded", ck)

d = np.load(os.path.expandvars(cfg.train_dataset_path)) if hasattr(cfg, "train_dataset_path") else \
    np.load("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/dice_rl_official/data_dir/robomimic/square/train.npz")
states = torch.tensor(d["states"][:512, :1] if d["states"].ndim == 3 else d["states"][:512],
                      dtype=torch.float32).cuda()
acts = torch.tensor(d["actions"][:512], dtype=torch.float32).cuda()
if acts.dim() == 2: acts = acts.reshape(len(acts), -1)
S = acts.shape[-1]
crit = model.critic if hasattr(model, "critic") else model.critic_q
def Q(s, a):
    with torch.no_grad():
        out = crit({"state": s} if isinstance(s, torch.Tensor) else s, a)
        q = torch.stack(list(out), 0).mean(0) if isinstance(out, (tuple, list)) else out
        return q.squeeze(-1)
a0 = acts.clone().requires_grad_(True)
q0 = Q(states, a0); q0.sum().backward()
gdir = a0.grad / a0.grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
radii = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
rows = []
for r in radii:
    qg = Q(states, (acts + r * np.sqrt(S) * gdir).detach()).mean().item()
    rnd = torch.randn_like(acts); rnd = rnd / rnd.norm(dim=-1, keepdim=True)
    qr = Q(states, (acts + r * np.sqrt(S) * rnd).detach()).mean().item()
    rows.append((r, qg, qr))
    print(f"rms_dist={r:5.2f}  Q_along_own_gradient={qg:9.4f}  Q_random_dir={qr:9.4f}")
np.save(OUT, np.array(rows))
print("saved", OUT)
