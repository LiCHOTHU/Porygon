#!/usr/bin/env python3
"""Stitch three raw rollout clips (early / mid / final) into one captioned
learning-progression video. CPU only (imageio). Run in the `porygon` env.

Usage: python stitch_learning.py OUT.mp4 TITLE early=CLIP mid=CLIP final=CLIP \
                                 [--sub "early sub|mid sub|final sub"]
"""
import sys, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont

args = sys.argv[1:]
OUT, TITLE = args[0], args[1]
stage_clips = {}
subs = ["", "", ""]
i = 2
while i < len(args):
    a = args[i]
    if a == "--sub":
        subs = args[i + 1].split("|"); i += 2; continue
    k, v = a.split("=", 1); stage_clips[k] = v; i += 1

STAGES = [("early", "EARLY", subs[0]), ("mid", "MID-TRAINING", subs[1]),
          ("final", "FINAL (converged)", subs[2])]

def fonts():
    try:
        return (ImageFont.truetype("DejaVuSans-Bold.ttf", 30),
                ImageFont.truetype("DejaVuSans.ttf", 20))
    except Exception:
        return None, None
FB, FS = fonts()

def read(path):
    r = imageio.get_reader(path); frames = [f for f in r]; r.close(); return frames

# target size from the first available clip
first = next(read(stage_clips[s[0]]) for s in STAGES if s[0] in stage_clips)
H, W = first[0].shape[:2]

def banner(frame, big, small):
    im = Image.fromarray(frame[:, :, :3]).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 62], fill=(0, 0, 0))
    d.text((14, 6), big, fill=(255, 255, 255), font=FB)
    d.text((14, 40), small, fill=(150, 210, 255), font=FS)
    return np.asarray(im)

writer = imageio.get_writer(OUT, fps=20, codec="libx264", quality=8, macro_block_size=8)
HOLD = 12  # freeze frames between stages
for key, label, sub in STAGES:
    if key not in stage_clips:
        continue
    frames = read(stage_clips[key])
    cap = f"{TITLE}  —  {label}"
    for f in frames:
        writer.append_data(banner(f, cap, sub))
    # brief hold on the last frame so the stage reads
    for _ in range(HOLD):
        writer.append_data(banner(frames[-1], cap, sub))
    print(f"stage {label}: {len(frames)} frames  ({sub})", flush=True)
writer.close()
print("WROTE", OUT, flush=True)
