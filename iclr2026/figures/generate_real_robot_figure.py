"""Extract documented keyframes from recorded teleop demonstrations for the paper."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEQUENCES = [
    {"task": "jigglypuff", "take": "20260910_162109",
     "title": "(a) Jigglypuff into the case", "frames": [0, 64, 97, 129, 159]},
    {"task": "stack", "take": "20260910_165952",
     "title": "(b) Red cube onto the purple cube", "frames": [0, 78, 118, 140, 160]},
]
STAGES = ["Initial", "Grasp", "Transfer", "Align", "Release"]
CROP = (0, 70, 640, 480)  # Same top-background crop for every panel; no aspect distortion.


def extract(video, indices, destination):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    selected = []
    try:
        for index in indices:
            if not 0 <= index < count:
                raise ValueError(f"Frame {index} outside {video} ({count} frames)")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index} of {video}")
            cv2.imwrite(str(destination / f"{video.stem}_frame_{index:04d}.png"), frame)
            selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return selected, {"video": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                      "frame_count": count, "fps": fps, "selected_indices_zero_based": indices,
                      "nominal_times_s": [round(i / fps, 3) for i in indices]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/pair/Desktop/openarm_exp/openarm_ws"))
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    frames_dir = output / "real_robot_frames"
    frames_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(12, 8.1), facecolor="white")
    grid = fig.add_gridspec(5, 5, height_ratios=[1, 1, .65, 1, 1],
                            left=.065, right=.995, bottom=.04, top=.91,
                            hspace=.09, wspace=.025)
    manifest = {"kind": "successful teleoperated task illustrations, not autonomous evaluation",
                "alignment": "same nominal video index at 15 fps; no per-frame capture timestamps available",
                "crop_xyxy": list(CROP), "stages": STAGES, "sequences": []}
    for task_index, spec in enumerate(SEQUENCES):
        directory = args.data_root / f"data_collection_{spec['task']}" / spec["take"]
        sequence = dict(spec, sources={})
        for view_index, (view, label) in enumerate((("chest", "Chest"), ("left_wrist", "Wrist"))):
            matches = sorted(directory.glob(view + "_rgb_*.avi"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {view} video in {directory}")
            frames, source = extract(matches[0], spec["frames"], frames_dir)
            if abs(source["fps"] - 15) > .01:
                raise ValueError("Selected figure indices assume a 15 fps recording")
            sequence["sources"][view] = source
            row = task_index * 3 + view_index
            for col, frame in enumerate(frames):
                ax = fig.add_subplot(grid[row, col])
                x0, y0, x1, y1 = CROP
                ax.imshow(frame[y0:y1, x0:x1])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#d5d5d5")
                    spine.set_linewidth(.6)
                if col == 0:
                    ax.set_ylabel(label, fontsize=14, labelpad=8)
                if view_index == 0:
                    ax.set_title(STAGES[col], fontsize=13, pad=4)
                else:
                    ax.set_xlabel(f"{spec['frames'][col] / 15:.1f} s", fontsize=11, labelpad=3)
                if col == 0 and view_index == 0:
                    pos = ax.get_position()
                    fig.text(.065, pos.y1 + .042, spec["title"], fontsize=15,
                             fontweight="bold", color="#19384b", va="bottom")
        manifest["sequences"].append(sequence)
    fig.savefig(output / "real_robot_tasks.pdf", dpi=300, bbox_inches="tight", pad_inches=.03)
    fig.savefig(output / "real_robot_tasks.png", dpi=160, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    (output / "real_robot_tasks.json").write_text(json.dumps(manifest, indent=2))
    print(output / "real_robot_tasks.pdf")


if __name__ == "__main__":
    main()
