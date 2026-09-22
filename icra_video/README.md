# CAST — ICRA supplementary-video slides

A **10-slide, 16:9, three-minute animated draft** based on `_ICRA_2027__CAST`.

## Open these files

- **`cast_video_slides.pptx`** — editable PowerPoint. Text, diagrams, bars, and placeholder frames are native objects. Five method clips are embedded MP4s with automatic-start timing; the residual equation and photographs are image assets. Each slide has a timed narration in its speaker notes and an automatic advance time.
- **`cast_video_slides.pdf`** — matching layout preview with animation poster frames.
- **`cast_method_preview.mp4`** — the first two minutes with animations, rendered at 1080p; silent, ready for narration review.
- **`animations/index.html`** — full-size player for the five individual clips.
- **`preview.html`** — local browser preview with synchronized video overlays and a scrubber. Arrow keys change slides; Space plays/pauses the timed 180-second sequence. The Notes button displays the narration. It works directly from disk, without a server or external libraries.
- **`previews/contact_sheet.jpg`** — all slides at a glance.
- **`narration.md`** — full script, timing table, source references, and media cues.

## Structure

| Time | Slides | Story |
|---|---|---|
| 00:00–00:30 | 1–2 | A mostly correct robot skill can still fail; small critic-guided updates can accumulate. |
| 00:30–01:32 | 3–5 | Frozen base + residual; same-noise pairing; critic proposal, weak/radial restoration, combined cap; fixed-target fitting and the penalty interpretation. |
| 01:32–02:00 | 6 | Recorded steps-to-success evidence and the restoration ablation; animated measured ablation comparison. |
| 02:00–02:16 | 7 | OpenArm, chest/wrist RGB, joint/gripper state, and offline refinement before autonomous evaluation. |
| 02:16–02:52 | 8–9 | Placement and stacking, with autonomous-video placeholders and reported success rates. |
| 02:52–03:00 | 10 | Improve the skill while keeping a persistent reference. |

**Method/idea/evidence: 120 seconds. Robot/takeaway: 60 seconds.**

## Completed animations

Five clips (**A01–A05**) show policy-distribution drift, paired residual actions, the CAST target construction, actual actor regression, and the measured LIBERO ablation. They are embedded in the PowerPoint and browser preview. Two autonomous-video slots (**V01–V02**) remain reserved.

- `animation_manifest.json` gives exact placement, clip paths, posters, and completion status.
- `animations/README.md` explains the teaching sequence, CS 285 reference, toy-model settings, exact equations, and benchmark-footage plan.
- `animations/toy_trace.json` records the actual actor parameter updates used in the animation.
- `scripts/build_animations.py` implements training and rendering. No robot learning curves are invented.
- `assets/source_manifest.json` identifies the source of photographs, result data, and teaching references.

The MP4 and browser preview are the most reliable ways to review playback here. Native PowerPoint playback was unavailable, although embedded-video relationships and automatic-start XML are validated. The PDF is static.

## Data and media choices in this draft

- The submitted paper's numerical measurements are unchanged. Figure summaries support mean steps to the first evaluation reaching 90% success; they do not provide a complete training curve. A05 animates the exact five-task component aggregates instead. Full learning curves can be added if the original evaluation logs become available.
- Robot photographs are labeled **teleoperated demonstration stills**, not autonomous CAST executions. V01/V02 need correctly attributed autonomous footage before final export.
- The submitted main text and supplement disagree on stacking pretraining and evaluation counts (80/30 versus approximately 60/20). Those counts are omitted pending author confirmation. Reported success percentages match the paper's figure and data file.
- The first draft explains the restoration penalty and local-gradient interpretation. It does not reproduce the full Wasserstein proof. The animation brief records that the distributional bound concerns candidate actions before selection and is conditional on the actual expected residual penalty.

## Rebuild

Use Python 3 with the packages in `requirements.txt` FFmpeg for video, and Poppler's `pdftoppm` for slide previews:

```bash
python -m pip install -r icra_video/requirements.txt
python icra_video/scripts/build_animations.py
python icra_video/scripts/build_slides.py
python icra_video/scripts/build_preview.py
python icra_video/scripts/render_video_preview.py
python icra_video/scripts/validate_deck.py
python icra_video/scripts/validate_animations.py
```

The current environment also supports temporary dependencies in `/tmp/icra-video-deps`, which the scripts detect automatically. Override that directory with `ICRA_VIDEO_DEPS` if needed. Use `--no-preview` to build the deck and PDF without Poppler.

The deck uses **DejaVu Sans**. Use the same font when editing or recording to retain the previewed typography. Editable content lives in the PowerPoint; reproducible layouts are in `scripts/build_slides.py`, and timings/narration are in `storyboard.json`. Rebuilding replaces generated PPTX/PDF outputs, so retain a separate copy of direct PowerPoint edits.

All new files stay in `icra_video`; the submitted paper is unchanged.
