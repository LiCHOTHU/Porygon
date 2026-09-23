#!/usr/bin/env python3
"""Unify learning clips into one consistent form: every stage fit into a common
CANVAS square with an identical caption bar, concatenated. CPU only (porygon).

Usage:
  python unified_stitch.py OUT.mp4 TITLE  LABEL::SUB::clip.mp4  LABEL::SUB::clip.mp4 ...
"""
import sys, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont

OUT, TITLE = sys.argv[1], sys.argv[2]
STAGES = [a.split("::", 2) for a in sys.argv[3:]]   # [LABEL, SUB, path]

CANVAS = 512          # square video area
BAR = 60              # caption bar height
FPS = 20
HOLD = 14             # freeze frames between stages
try:
    FB = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    FS = ImageFont.truetype("DejaVuSans.ttf", 19)
    FT = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
except Exception:
    FB = FS = FT = None

def read(p):
    r = imageio.get_reader(p); f = [x for x in r]; r.close(); return f

def frame(img, big, small):
    im = Image.fromarray(img[:, :, :3]).convert("RGB")
    # scale (up OR down) to fit CANVAS, preserve aspect, then center on black
    f = min(CANVAS / im.width, CANVAS / im.height)
    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    canvas = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    canvas.paste(im, ((CANVAS - im.width) // 2, (CANVAS - im.height) // 2))
    out = Image.new("RGB", (CANVAS, CANVAS + BAR), (0, 0, 0))
    out.paste(canvas, (0, BAR))
    d = ImageDraw.Draw(out)
    d.text((14, 6), big, fill=(255, 255, 255), font=FB)
    d.text((14, 38), small, fill=(150, 210, 255), font=FS)
    # small persistent title tag, top-right
    if FT is not None:
        w = d.textlength(TITLE, font=FT)
        d.text((CANVAS - w - 12, 10), TITLE, fill=(120, 120, 120), font=FT)
    return np.asarray(out)

w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
for label, sub, path in STAGES:
    frs = read(path)
    for fr in frs:
        w.append_data(frame(fr, label, sub))
    for _ in range(HOLD):
        w.append_data(frame(frs[-1], label, sub))
    print(f"stage {label}: {len(frs)} frames", flush=True)
w.close()
print("WROTE", OUT, CANVAS, "x", CANVAS + BAR, flush=True)
