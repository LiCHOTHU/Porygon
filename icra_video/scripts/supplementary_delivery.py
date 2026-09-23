#!/usr/bin/env python3
"""Export the final cut's scene, narration, and source-media timing together."""
import csv
import html
from pathlib import Path

FPS = 30


def clock(seconds):
    return f'{int(seconds)//60:02}:{int(seconds)%60:02}'


def timecode(seconds):
    frame = round(seconds * FPS)
    return f'{frame//108000:02}:{frame//1800%60:02}:{frame//30%60:02}:{frame%30:02}'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write(story, out):
    root = out.parent
    duration = story['duration_seconds']
    scene_rows, cue_rows, source_rows = [], [], []
    previous_end = 0
    for s in story['slides']:
        start, end = s['start'], s['start'] + s['duration']
        assert start == previous_end
        previous_end = end
        scene_rows.append(dict(scene=s['id'],title=s['section'],start=clock(start),end=clock(end),
                               start_seconds=start,end_seconds=end,duration_seconds=s['duration'],
                               start_frame=round(start*FPS),end_frame_exclusive=round(end*FPS),
                               timecode_in=timecode(start),timecode_out_exclusive=timecode(end),
                               segment=s['segment']))
        for cue in s['voiceover_cues']:
            a,b=start+cue['start'],start+cue['end']
            cue_rows.append(dict(cue=len(cue_rows)+1,scene=s['id'],start_seconds=a,end_seconds=b,
                                 timecode_in=timecode(a),timecode_out_exclusive=timecode(b),
                                 start_frame=round(a*FPS),end_frame_exclusive=round(b*FPS),
                                 visual=cue['visual'],spoken_text=cue['text']))
        def source(label,path,source_fps,a,b,speed,policy):
            source_rows.append(dict(scene=s['id'],panel=label,output_start_seconds=start,output_end_seconds=end,
                                    output_start_frame=round(start*FPS),output_end_frame_exclusive=round(end*FPS),
                                    source_path=path,source_fps=source_fps,source_start_seconds=a,source_end_seconds=b,
                                    source_start_frame=round(a*source_fps),source_end_frame_exclusive=round(b*source_fps),
                                    output_motion_end_seconds=start+(b-a)/speed,
                                    last_frame_hold_seconds=max(0,s['duration']-(b-a)/speed),
                                    speed_relative_to_source=speed,policy_or_content=policy))
        if 'simulation_source' in s:
            r=s['simulation_source']
            for stage in r['stages']:
                source(stage['label'],r['source'],r['actual_source_fps'],stage['source_start_seconds'],
                       stage['source_end_seconds'],stage['speed_relative_to_source'],r['policy'])
        elif 'rollout_source' in s:
            r=s['rollout_source']
            for camera,stream in r['streams'].items():
                source(camera,stream['path'],15,r['source_start_seconds'],
                       r['source_start_seconds']+r['source_duration_seconds'],r['playback_speed'],r['policy_label'])
        elif s['kind']=='existing':
            a=s.get('source_trim_start',0)
            source('Full scene',s['source_segment'],FPS,a,a+s['duration'],1,s['section'])
        else:
            source('Static slide',str(out/'assets'/f"{s['kind']}.png"),FPS,0,s['duration'],1,s['section'])
    assert previous_end == duration <= 180
    assert cue_rows[0]['start_seconds']==0 and cue_rows[-1]['end_seconds']==duration
    assert all(a['end_seconds']==b['start_seconds'] for a,b in zip(cue_rows,cue_rows[1:]))
    write_csv(out/'timeline.csv',scene_rows)
    write_csv(out/'voiceover_timing.csv',cue_rows)
    write_csv(out/'source_timing.csv',source_rows)
    table=['| Scene | Start–end | Duration | Content |','|---|---|---|---|']
    table += [f"| {r['scene']} | {r['start']}–{r['end']} | {r['duration_seconds']} s | {r['title']} |" for r in scene_rows]
    timing_notes='''## Recording against this cut

- Read the [continuous script](voiceover_script.md); its 14 paragraphs match the 14 scenes in the timeline. Use the [phrase-by-phrase cue sheet](voiceover_cue_sheet.md) to time each sentence. Only the script text is spoken.
- Record one scene at a time if helpful. Place each recording at that scene's start; leave natural pauses within its allotted duration. The file `voiceover_timing.csv` gives precise phrase boundaries for editing.
- Figure 2 remains at **00:26–00:56**. The critic gradient is capped at 00:34–00:37; the combined target step is capped around the current action at 00:44–00:50.
- Robot action cues follow the visible grasp, transport/alignment, and release. Both robot clips use continuous 11-second excerpts at normal playback speed.
- Simulation stages play side by side. Their speed labels refer to the supplied videos. Shorter stages hold their completed final frame while longer stages finish playing. Exact source intervals, speed factors, and hold durations are in `source_timing.csv`.
- All times are absolute from the first video frame. The video uses constant **30 fps** and non-drop-frame `HH:MM:SS:FF` timecode. CSV frame ranges are zero-based and end-exclusive: a scene ending at frame 180 is immediately followed by the next scene starting at frame 180.
- The master and captioned rehearsal have identical timing. Narration has not yet been recorded; subtitles are delivery cues, not measured speech timings. The cut is 2:59, leaving 1 second within the three-minute limit.
'''
    (out/'timeline.md').write_text('# CAST — final video timing\n\n**02:59 · 14 scenes · 30 fps · 5,370 frames.**\n\n'+ '\n'.join(table)+'\n\n'+timing_notes)
    readme='''# CAST — supplementary video

**2:59 · 1920 × 1080 · 30 fps · within the three-minute limit.**

- [Captioned rehearsal video](cast_supplementary_video_captioned.mp4)
- [Clean editing master](cast_supplementary_video.mp4), with optional English subtitles
- [Video player with scene navigation](index.html)
- [Timestamped reading script](voiceover_recording_script.md), [continuous spoken script](voiceover_script.md), and [recording cues](voiceover_cue_sheet.md)
- [Exact scene timeline](timeline.md): [scene CSV](timeline.csv), [phrase CSV](voiceover_timing.csv), [source-media CSV](source_timing.csv)
- [Subtitle file](captions.srt) and [selected frames](contact_sheet.jpg)

## Edit

'''+ '\n'.join(table)+'''

## Footage and attribution

**LIBERO task 32** compares the frozen base with CAST from the same starting state. **robomimic square** compares three CAST checkpoints. **DMC walker-run** shows three CAST training checkpoints. Original stage intervals play side by side, with speed factors displayed. Shorter stages play at source speed and hold their final frames so their outcomes remain visible; longer stages are accelerated to fit. The crop removes source title banners and letterbox bars while retaining the complete simulator image. `simulation_sources.json` records source metadata, actual encoded frame counts, stage boundaries, and hashes.

Both **robot rollouts use CAST-trained policies**, as confirmed by the author. The successful placement and stacking examples show chest and wrist images from the same run at **1× speed**. Each uses an 11-second continuous excerpt starting at source frame zero. Visual inspection confirms release into the case and release of the red cube onto the purple cube. The paper's CAST evaluation results appear on a separate results slide. `rollout_sources.json` records source paths, policy attribution, trims, and hashes. Original recordings and the submitted paper are unchanged.

Narration audio has not been recorded. The script, captions, and timing files follow this exact edit. The root `icra_video/voiceover_script.md` is the same reading copy. The original ten-slide deck and seven-scene paper-only cut remain earlier production assets; use this cut's 14-scene timeline for recording.

## Rebuild

```bash
python icra_video/scripts/build_supplementary_video.py
```

Requires the paper-only segments in `../explanation/segments/`, the three source clips and JSON sidecars in `../task_demos/`, and the selected recordings in `/home/licho/workspace/real_policy_eval`. `validation.json` records dimensions, exact duration and frame count, checksums, and full decode checks.
'''
    (out/'README.md').write_text(readme)
    chapters=''.join(f'<button onclick="seek({r["start_seconds"]})">{r["start"]} · {html.escape(r["title"])}</button>' for r in scene_rows)
    player='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST supplementary video</title><style>:root{background:white;color:#202124;font-family:system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}video{width:100%;aspect-ratio:16/9}p{color:#626b73;line-height:1.6}a{color:#287764}button{font:inherit;background:white;border:1px solid #d5d9dd;padding:8px 12px;margin:0 8px 10px 0;cursor:pointer}.chapters{display:flex;flex-wrap:wrap}</style><main><h1>CAST supplementary video</h1><p>2:59 · 1080p · 14 scenes · ready for your narration.</p><button onclick="pick(true)">Captioned rehearsal</button><button onclick="pick(false)">Clean master</button><video id="v" src="cast_supplementary_video_captioned.mp4" poster="frames/047.jpg" controls playsinline preload="metadata"></video><p><a href="voiceover_recording_script.md">Timestamped reading script</a> · <a href="voiceover_cue_sheet.md">Recording cues</a> · <a href="timeline.md">Exact timeline</a> · <a href="captions.srt">Subtitles</a> · <a href="README.md">Sources and editing notes</a></p><h2>Scene navigation</h2><div class="chapters">'''+chapters+'''</div><p>The simulations show CAST learning across LIBERO, robomimic, and DMC. The robot rollouts show successful CAST placement and stacking, followed by aggregate evaluation results.</p></main><script>function pick(c){const v=document.getElementById('v'),t=v.currentTime;v.pause();v.src=c?'cast_supplementary_video_captioned.mp4':'cast_supplementary_video.mp4';v.onloadedmetadata=()=>v.currentTime=Math.min(t,v.duration)}function seek(t){document.getElementById('v').currentTime=t}</script></html>'''
    (out/'index.html').write_text(player)
    cue_path=out/'voiceover_cue_sheet.md'
    text=cue_path.read_text().split('\n## Recording against this cut')[0]
    cue_path.write_text(text.rstrip()+'\n\n'+timing_notes+'\nExact scene table: [timeline.md](timeline.md).\n')
    reading=['# CAST — timestamped reading script', '',
             '**Final cut: 02:59.** Read only the paragraphs aloud. Headings give the exact scene times; '
             'pause naturally within each interval. For sentence-level timing, use [the cue sheet](voiceover_cue_sheet.md).', '']
    for scene in story['slides']:
        reading += [f"## {clock(scene['start'])}–{clock(scene['start']+scene['duration'])} · {scene['section']}", '', scene['narration'], '']
    (out/'voiceover_recording_script.md').write_text('\n'.join(reading))
    (root/'voiceover_recording_script.md').write_text('\n'.join(reading))
    (root/'voiceover_script.md').write_text((out/'voiceover_script.md').read_text())
    cue_text=cue_path.read_text().replace('(timeline.md)','(supplementary/timeline.md)')
    (root/'voiceover_cue_sheet.md').write_text(cue_text+'\nVideo: [captioned cut](supplementary/cast_supplementary_video_captioned.mp4). '
                                            'This edit is defined in `supplementary/edit_manifest.json`; CSV timing files are in `supplementary/`.\n')
