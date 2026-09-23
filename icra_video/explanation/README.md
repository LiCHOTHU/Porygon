# CAST — paper explanation video

**1:44 · 1920 × 1080 · 30 fps.** White academic layouts, dark typography, restrained component colors, and no decorative cards or progress bars. This standalone cut covers the idea, animated method, simulation evidence, and takeaway.

- [Captioned rehearsal video](cast_paper_explanation_captioned.mp4): captions sit below the slides so they do not cover the math. Ready to rehearse or record your voice against.
- [Clean editing master](cast_paper_explanation.mp4): full-size visuals, optional English subtitle track, and seven chapters. No recorded narration yet.
- [Spoken script](voiceover_script.md) and [recording cues](voiceover_cue_sheet.md): seven paragraphs, aligned to this cut.
- [SRT captions](captions.srt): import into a video editor.
- [Contact sheet](contact_sheet.jpg): selected frames from the exported video.
- `segments/`: seven separately rendered scenes for the next edit.
- `edit_manifest.json`: source slides, timings, and exact media paths.

## Later footage

The original ten-slide storyboard is retained. This cut uses slides 1–6 and the closing slide, with no simulation or robot rollout clips. The opening slide retains its labeled robot photograph. A benchmark overview card replaces the earlier task montage; the narration has been adjusted to match.

**76 seconds remain within the three-minute limit.** Insert your simulation and robot material at **01:36**, before the eight-second closing scene. The original robot setup/results narration remains in `../voiceover_script.md` for that edit. This folder's script follows only the exported explanation cut.

## Rebuild

```bash
python icra_video/scripts/build_explanation_video.py
```

The export uses the existing code-generated method animations and the submitted paper's result data. Figure 2 keeps its existing 30-second timing. The measured chart is an aggregate comparison, not a fabricated learning curve. `validation.json` records frame counts, durations, checksums, and completed decode checks.
