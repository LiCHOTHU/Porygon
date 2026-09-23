# CAST — final video timing

**02:59 · 14 scenes · 30 fps · 5,370 frames.**

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

## Recording against this cut

- Read the [continuous script](voiceover_script.md); its 14 paragraphs match the 14 scenes in the timeline. Use the [phrase-by-phrase cue sheet](voiceover_cue_sheet.md) to time each sentence. Only the script text is spoken.
- Record one scene at a time if helpful. Place each recording at that scene's start; leave natural pauses within its allotted duration. The file `voiceover_timing.csv` gives precise phrase boundaries for editing.
- Figure 2 remains at **00:26–00:56**. The critic gradient is capped at 00:34–00:37; the combined target step is capped around the current action at 00:44–00:50.
- Robot action cues follow the visible grasp, transport/alignment, and release. Both robot clips use continuous 11-second excerpts at normal playback speed.
- Simulation stages play side by side. Their speed labels refer to the supplied videos. Shorter stages hold their completed final frame while longer stages finish playing. Exact source intervals, speed factors, and hold durations are in `source_timing.csv`.
- All times are absolute from the first video frame. The video uses constant **30 fps** and non-drop-frame `HH:MM:SS:FF` timecode. CSV frame ranges are zero-based and end-exclusive: a scene ending at frame 180 is immediately followed by the next scene starting at frame 180.
- The master and captioned rehearsal have identical timing. Narration has not yet been recorded; subtitles are delivery cues, not measured speech timings. The cut is 2:59, leaving 1 second within the three-minute limit.
