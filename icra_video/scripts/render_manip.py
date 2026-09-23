#!/usr/bin/env python3
"""Render full-length MP4 clips of the manipulation tasks CAST is evaluated on,
by replaying a successful demonstration through the actual simulator.

These are task-illustration clips (exact simulator, camera and successful
trajectory of the benchmark task). They are demonstration replays, not CAST
policy rollouts -- labelled as such in each provenance file. Run in `porygon`.
"""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "task_demos"


class Writer:
    def __init__(self, path, w, h, fps):
        import imageio
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                    quality=8, macro_block_size=8)
        self.count = 0

    def write(self, frame):
        self.w.append_data(np.ascontiguousarray(frame)); self.count += 1

    def close(self):
        self.w.close()


def relocate_assets(xml):
    root = ET.fromstring(xml)
    suite = Path(importlib.util.find_spec("robosuite").origin).parent
    # libero is a namespace package (origin=None); use its search path or env var
    libero = None
    spec = importlib.util.find_spec("libero")
    if spec is not None:
        loc = spec.origin or (list(spec.submodule_search_locations) or [None])[0]
        if loc:
            libero = Path(loc).parent / "libero" if spec.origin else Path(loc) / "libero"
    changed = 0
    for el in root.iter():
        old = el.get("file")
        if not old:
            continue
        new = old
        if "/robosuite/" in old:
            new = str(suite / "models/assets" / old.split("/models/assets/", 1)[1])
        elif "/assets/" in old:
            rel = old.split("/assets/", 1)[1]
            roots = [p for p in [libero / "assets" if libero else None,
                                 Path(os.environ.get("ICRA_LIBERO_ASSETS", "/none"))] if p]
            cands = [r / rel for r in roots]
            new = str(next((p for p in cands if p.exists()), cands[0] if cands else old))
        if not Path(new).exists():
            raise FileNotFoundError(new)
        el.set("file", new); changed += old != new
    return ET.tostring(root, encoding="unicode"), changed


def replay(label, path, demo, camera, limit=None):
    import h5py, mujoco
    with h5py.File(path, "r") as f:
        group = f["data"][demo]
        states = group["states"][:]
        xml, reloc = relocate_assets(group.attrs["model_file"])
        rewards = group["rewards"][:] if "rewards" in group else None
    tree = ET.fromstring(xml); compat = []
    for opt in tree.findall("option"):
        if "collision" in opt.attrib:
            compat.append("removed obsolete option/collision"); del opt.attrib["collision"]
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    model.vis.global_.offwidth = 1280; model.vis.global_.offheight = 720
    expected = 1 + model.nq + model.nv + model.na
    assert states.shape[1] == expected, (states.shape, expected)
    fps = 20
    renderer = mujoco.Renderer(model, height=720, width=1280)
    vis = mujoco.MjvOption(); vis.geomgroup[0] = 0; vis.geomgroup[1] = 1; vis.sitegroup[:] = 0
    out = OUT / f"{label}.mp4"; writer = Writer(out, 1280, 720, fps)
    seq = states[:limit] if limit else states
    for i, state in enumerate(seq):
        data.time = state[0]; data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
        if model.na:
            data.act[:] = state[-model.na:]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=vis)
        writer.write(renderer.render())
        if i % 40 == 0:
            print(label, i, len(seq), flush=True)
    writer.close(); renderer.close()
    rec = {"clip": label, "kind": "successful demonstration replay in the benchmark simulator",
           "cast_evaluation": False, "policy_checkpoint": None,
           "source_dataset": str(path), "episode": demo, "camera": camera,
           "frames": writer.count, "fps": fps, "duration_seconds": round(writer.count / fps, 2),
           "resolution": [1280, 720], "mujoco_version": mujoco.__version__,
           "asset_paths_relocated": reloc, "xml_compat_edits": compat,
           "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16],
           "source_final_reward": float(rewards[-1]) if rewards is not None else None}
    (OUT / f"{label}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("WROTE", out, writer.count, "frames", flush=True)


CLIPS = {
    "libero_red_mug": dict(
        path="data/libero/libero_90_unprocessed/"
             "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5",
        demo="demo_0", camera="agentview"),
    "libero_ketchup": dict(
        path="data/libero/libero_90_unprocessed/"
             "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_demo.hdf5",
        demo="demo_0", camera="agentview"),
    "robomimic_square": dict(
        path="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/imitation_scratch/"
             "mimicgen_data/core/square_d0.hdf5",
        demo="demo_0", camera="agentview"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.clips:
        c = CLIPS[name]
        try:
            replay(name, c["path"], c["demo"], c["camera"], a.limit)
        except Exception as e:
            print("FAILED", name, repr(e), flush=True)
