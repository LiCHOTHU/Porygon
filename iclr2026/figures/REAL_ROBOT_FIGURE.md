# Real-robot task figure

`real_robot_tasks.pdf` / `.png` illustrate two successful **teleoperated**
demonstrations. They are not CAST, DICE-RL, refined-BC, or autonomous base-policy
evaluation results, and do not fill the manuscript success-rate table.

Sources under `/home/pair/Desktop/openarm_exp/openarm_ws/`:

| Task | Take | Zero-based frame indices in each RGB view |
|---|---|---|
| Jigglypuff into case | `data_collection_jigglypuff/20260910_162109` | 0, 64, 97, 129, 159 |
| Red-on-purple stacking | `data_collection_stack/20260910_165952` | 0, 78, 118, 140, 160 |

Both selected take pairs have equal chest/wrist video frame counts and 15 fps
headers. Nominal frame-index correspondence is used; the recordings do not
retain per-frame acquisition timestamps, so hardware synchronization is not
claimed. The sequences were inspected through release/end: the toy is visible
inside the case, and the cube stays on its support while the gripper withdraws.
The earlier release frame is selected for stacking so the wrist still sees the
completed stack rather than looking away during withdrawal.

The figure crops the top 70 background pixels uniformly and preserves aspect
ratio. Individual full-frame PNGs are in `real_robot_frames/`. Source paths,
SHA-256 hashes, frame counts, indices, nominal times, and crop coordinates are
recorded in `real_robot_tasks.json`.

Regenerate from the Porygon repository:

```bash
.venv/bin/python iclr2026/figures/generate_real_robot_figure.py
```

Use `--data-root` if the OpenArm workspace is located elsewhere. Requires OpenCV
and matplotlib. The original recordings are read-only inputs.
