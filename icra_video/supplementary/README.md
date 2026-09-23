# CAST — supplementary video

**2:59 · 1920 × 1080 · 30 fps · within the three-minute limit.**

- [Captioned rehearsal video](cast_supplementary_video_captioned.mp4)
- [Clean editing master](cast_supplementary_video.mp4), with optional English subtitles
- [Video player with scene navigation](index.html)
- [Timestamped reading script](voiceover_recording_script.md), [continuous spoken script](voiceover_script.md), and [recording cues](voiceover_cue_sheet.md)
- [Exact scene timeline](timeline.md): [scene CSV](timeline.csv), [phrase CSV](voiceover_timing.csv), [source-media CSV](source_timing.csv)
- [Subtitle file](captions.srt) and [selected frames](contact_sheet.jpg)

## Edit

| Scene | Start–end | Duration | Content |
|---|---|---|---|
| 01 | 00:00–00:06 | 6 s | The idea |
| 02 | 00:06–00:18 | 12 s | Why anchoring matters |
| 03 | 00:18–00:26 | 8 s | Policy architecture |
| 04 | 00:26–00:56 | 30 s | Method and equations |
| 05 | 00:56–01:06 | 10 s | Learn and repeat |
| 06 | 01:06–01:17 | 11 s | LIBERO: place the bottle in the drawer |
| 07 | 01:17–01:27 | 10 s | robomimic: learning square-nut insertion |
| 08 | 01:27–01:38 | 11 s | DMC walker-run: locomotion learning |
| 09 | 01:38–01:59 | 21 s | Simulation evidence |
| 10 | 01:59–02:15 | 16 s | Real robot setup |
| 11 | 02:15–02:26 | 11 s | Successful placement example |
| 12 | 02:26–02:37 | 11 s | Successful stacking example |
| 13 | 02:37–02:51 | 14 s | Reported robot results |
| 14 | 02:51–02:59 | 8 s | Takeaway |

## Footage and attribution

**LIBERO task 32** compares the frozen base with CAST from the same starting state. **robomimic square** compares three CAST checkpoints. **DMC walker-run** shows three CAST training checkpoints. Original stage intervals play side by side, with speed factors displayed. Shorter stages play at source speed and hold their final frames so their outcomes remain visible; longer stages are accelerated to fit. The crop removes source title banners and letterbox bars while retaining the complete simulator image. `simulation_sources.json` records source metadata, actual encoded frame counts, stage boundaries, and hashes.

Both **robot rollouts use CAST-trained policies**, as confirmed by the author. The successful placement and stacking examples show chest and wrist images from the same run at **1× speed**. Each uses an 11-second continuous excerpt starting at source frame zero. Visual inspection confirms release into the case and release of the red cube onto the purple cube. The paper's CAST evaluation results appear on a separate results slide. `rollout_sources.json` records source paths, policy attribution, trims, and hashes. Original recordings and the submitted paper are unchanged.

Narration audio has not been recorded. The script, captions, and timing files follow this exact edit. The root `icra_video/voiceover_script.md` is the same reading copy. The original ten-slide deck and seven-scene paper-only cut remain earlier production assets; use this cut's 14-scene timeline for recording.

## Rebuild

```bash
python icra_video/scripts/build_supplementary_video.py
```

Requires the paper-only segments in `../explanation/segments/`, the three source clips and JSON sidecars in `../task_demos/`, and the selected recordings in `/home/licho/workspace/real_policy_eval`. `validation.json` records dimensions, exact duration and frame count, checksums, and full decode checks.
