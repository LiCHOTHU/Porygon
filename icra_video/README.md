# CAST — ICRA supplementary video

## Final cut: 2:59

The final cut combines the CAST method animation, all three supplied simulation videos, successful robot task examples, and the paper's measured results. It uses a white academic layout at **1920 × 1080 / 30 fps**.

- **[Captioned rehearsal video](supplementary/cast_supplementary_video_captioned.mp4)**
- **[Clean editing master](supplementary/cast_supplementary_video.mp4)**, with optional subtitles
- [Video player with scene navigation](supplementary/index.html)
- **[Timestamped reading script](voiceover_recording_script.md)** — one paragraph per scene, ready to read aloud
- [Continuous spoken script](voiceover_script.md) and [phrase-by-phrase recording cues](voiceover_cue_sheet.md)
- **[Exact timeline](supplementary/timeline.md)** — 14 scenes, 179 seconds, 5,370 frames
- [Scene timing CSV](supplementary/timeline.csv), [narration timing CSV](supplementary/voiceover_timing.csv), and [source-media timing CSV](supplementary/source_timing.csv)
- [Footage sources, attribution, and validation](supplementary/README.md)

Narration audio is ready to be recorded by the author. The scripts and subtitles follow this exact cut. The first 1:06 explains the method; simulation footage and results run to 1:59; the last minute covers the robot setup, successful task examples, results, and takeaway.

The simulations show CAST learning in LIBERO, robomimic, and DMC; LIBERO also includes a frozen-base comparison. The robot rollouts show CAST successfully completing placement and stacking. Aggregate results accompany the demonstrations. Source data and the submitted paper are unchanged.

### Rebuild the final video

```bash
python icra_video/scripts/build_supplementary_video.py
```

Requires Python dependencies in `requirements.txt`, FFmpeg, the paper-only scene segments in `explanation/segments/`, the supplied clips and metadata in `task_demos/`, and the selected robot recordings in `/home/licho/workspace/real_policy_eval`.

The final builder generates the videos, narration documents, subtitles, chapters, source manifests, and all timing sheets together. Use `supplementary/edit_manifest.json` to inspect this edit; the earlier `storyboard.json` describes the original slide deck.

## Earlier production assets

These remain useful for editing individual visuals. Their timelines precede the final 2:59 video above.

- [Paper-only academic cut](explanation/index.html): 1:44, seven scenes, method and simulation results; [rebuild notes](explanation/README.md).
- [Original PowerPoint](cast_video_slides.pptx) and [PDF layout preview](cast_video_slides.pdf): ten-slide draft with embedded method animations and earlier footage placeholders.
- [Original browser slide preview](preview.html) and [slide contact sheet](previews/contact_sheet.jpg).
- [Method animation documentation](animations/README.md) and [individual animation player](animations/index.html).
- [Original environment demonstrations](footage/index.html).

The Figure 2 animation uses the submitted figure's samples and vectors, with math beside each operation. Toy optimization animations explain the updates; aggregate charts use the submitted measurements. They do not invent robot learning curves. Source references are recorded in `assets/source_manifest.json`; experiment data are in `assets/experiment_data.json` and `assets/robot_results.json`.

To rebuild the earlier production assets:

```bash
python icra_video/scripts/build_animations.py
python icra_video/scripts/build_environment_montage.py
python icra_video/scripts/build_slides.py
python icra_video/scripts/build_preview.py
python icra_video/scripts/render_video_preview.py
python icra_video/scripts/validate_deck.py
python icra_video/scripts/validate_animations.py
```

The deck uses DejaVu Sans. Temporary slide dependencies can be provided with `ICRA_VIDEO_DEPS`. Rebuilding the earlier deck retains the final video's root reading script; its own slide notes still follow its original ten-slide storyboard.
