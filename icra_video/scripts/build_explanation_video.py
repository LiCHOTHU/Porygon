#!/usr/bin/env python3
"""Assemble the paper explanation, without simulation or robot rollout footage.

Exports an editing master, a captioned rehearsal cut, aligned narration, and
individual scene segments. Uses the existing submitted-data animations.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import os
import sys
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont
from academic_video_slides import build as build_academic_slides, POSITIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'explanation'
FPS = 30
BG, CARD, WHITE, MUTED, TEAL, BLUE = '#FFFFFF', '#FFFFFF', '#202124', '#626B73', '#287764', '#356A96'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def run(args):
    subprocess.run(args, check=True)


def ffmpeg(*args):
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)])


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-show_chapters',
        '-of', 'json', str(path)], text=True))


def stamp(t, separator=','):
    ms = round(t * 1000)
    return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}{separator}{ms % 1000:03}'


def clock(t):
    return f'{t // 60:02}:{t % 60:02}'


def prepare_story():
    original = json.loads((ROOT / 'storyboard.json').read_text())
    selected = copy.deepcopy(original['slides'][:6] + original['slides'][9:10])
    elapsed = 0
    for index, slide in enumerate(selected, 1):
        slide['source_slide_id'] = slide['id']
        slide['source_start'] = slide['start']
        slide['id'] = f'{index:02}'
        slide['start'] = elapsed
        elapsed += slide['duration']
    # This export has a benchmark overview card in place of the task montage.
    selected[5]['voiceover_cues'][0].update(
        text='We test this approach in LIBERO, robomimic, and DMC, covering manipulation and continuous control.',
        visual='Three benchmark settings: LIBERO images/sparse rewards; robomimic state/sparse rewards; DMC continuous control/dense rewards.')
    selected[5]['narration'] = ' '.join(c['text'] for c in selected[5]['voiceover_cues'])
    return dict(title='CAST — paper explanation', duration_seconds=elapsed,
                maximum_final_duration_seconds=180, remaining_footage_budget_seconds=180-elapsed,
                insert_rollouts_before_closing_at_seconds=selected[-1]['start'],
                audio='No narration recorded; ready for the author’s voiceover.',
                source_storyboard='../storyboard.json', slides=selected,
                footage='No simulation or real-robot rollout video is included. The opener retains the labeled robot demonstration photograph.')


def benchmark_card():
    im = Image.new('RGB', (1280, 720), CARD)
    d = ImageDraw.Draw(im)
    d.text((48, 40), 'Three evaluation settings', font=ImageFont.truetype(BOLD, 42), fill=WHITE)
    rows = [('LIBERO', 'Image observations · sparse rewards'),
            ('robomimic', 'State observations · sparse rewards'),
            ('DMC', 'Continuous control · dense rewards')]
    for i, (title, description) in enumerate(rows):
        y = 150 + i * 156
        d.line((48, y+137, 1232, y+137), fill='#D5D9DD', width=2)
        d.text((78, y+17), title, font=ImageFont.truetype(BOLD, 36), fill=WHITE)
        d.text((78, y+70), description, font=ImageFont.truetype(FONT, 29), fill=WHITE)
    d.text((48, 659), 'Manipulation and continuous control', font=ImageFont.truetype(FONT, 24), fill=MUTED)
    path = OUT / 'assets' / 'benchmark_overview.png'
    im.save(path)
    return path


def build_evidence_clip():
    path = OUT / 'assets' / 'A05_explanation_results.mp4'
    card = benchmark_card()
    source = OUT / 'animations' / 'A05_measured_learning_progress.mp4'
    ffmpeg('-loop', '1', '-framerate', FPS, '-i', card, '-i', source,
           '-filter_complex_threads', '2', '-filter_complex',
           '[0:v]trim=duration=9,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v0];'
           '[1:v]trim=duration=21,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v1];'
           '[v0][v1]concat=n=2:v=1:a=0[out]',
           '-map', '[out]', '-t', '30', '-an', '-r', FPS, '-c:v', 'libx264',
           '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '4', path)
    return path


def write_documents(story):
    srt, vtt = [], ['WEBVTT', '']
    ass = ['[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 1920', 'PlayResY: 1080',
           'WrapStyle: 2', 'ScaledBorderAndShadow: yes', '', '[V4+ Styles]',
           'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
           'Style: Default,DejaVu Sans,34,&H00242120,&H00242120,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,1,0,0,2,120,120,22,1',
           '', '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    script = [f"# {story['title']} — voiceover", '']
    cues = [f"# Recording cues — {story['title']}", '',
            f"**{clock(story['duration_seconds'])} · narration audio not yet recorded.** Read [voiceover_script.md](voiceover_script.md) "
            f"against the captioned video. The {len(story['slides'])} paragraphs follow its {len(story['slides'])} scenes. "
            'Cue times are delivery targets for your recording.', '',
            '| Time | Scene / on-screen event | Spoken text |', '|---|---|---|']
    cue_index = 0
    previous_end = 0
    for slide in story['slides']:
        script += [slide['narration'], '']
        for cue in slide['voiceover_cues']:
            cue_index += 1
            start, end = slide['start']+cue['start'], slide['start']+cue['end']
            assert previous_end == start < end <= story['duration_seconds']
            previous_end = end
            lines = textwrap.wrap(cue['text'], width=76, break_long_words=False, break_on_hyphens=False)
            assert len(lines) <= 2
            srt += [str(cue_index), f'{stamp(start)} --> {stamp(end)}', '\n'.join(lines), '']
            vtt += [f'{stamp(start,".")} --> {stamp(end,".")}', '\n'.join(lines), '']
            astart = f'{start//3600}:{start//60%60:02}:{start%60:02}.00'
            aend = f'{end//3600}:{end//60%60:02}:{end%60:02}.00'
            ass.append(f'Dialogue: 0,{astart},{aend},Default,,0,0,0,,'+r'\N'.join(lines))
            cues.append(f"| {clock(start)}–{clock(end)} | {cue['visual']} | {cue['text']} |")
    assert previous_end == story['duration_seconds']
    for name, lines in [('captions.srt', srt), ('captions.vtt', vtt), ('captions.ass', ass),
                        ('voiceover_script.md', script), ('voiceover_cue_sheet.md', cues)]:
        (OUT / name).write_text('\n'.join(lines)+'\n')
    (OUT / 'edit_manifest.json').write_text(json.dumps(story, indent=2)+'\n')
    metadata = [';FFMETADATA1', f"title={story['title']}"]
    for slide in story['slides']:
        metadata += ['[CHAPTER]', 'TIMEBASE=1/1000', f"START={slide['start']*1000}",
                     f"END={(slide['start']+slide['duration'])*1000}", f"title={slide['section']}"]
    (OUT / 'chapters.ffmetadata').write_text('\n'.join(metadata)+'\n')


def render_scene(slide, story, slots, evidence_clip):
    sid = int(slide['source_slide_id'])
    part = OUT / 'segments' / f"{slide['id']}_slide_{sid:02}.mp4"
    cmd = ['-loop', '1', '-framerate', FPS, '-i', OUT / 'posters' / f'slide-{sid:02}.png']
    active = [copy.deepcopy(m) for m in slots if m['slide'] == sid and m['status'].startswith('completed')]
    for m in active:
        m['filename'] = str(OUT / 'animations' / Path(m['filename']).name)
        if m['id'] == 'A05':
            m['filename'] = str(evidence_clip)
        source = ROOT / m['filename']
        assert source.is_file(), source
        cmd += ['-i', source]
    filters, last = [], '0:v'
    for i, m in enumerate(active, 1):
        x, y, w, h = POSITIONS[m['id']]
        w, h = w//2*2, h//2*2
        filters += [f'[{i}:v]scale={w}:{h}:flags=lanczos,setsar=1[clip{i}]',
                    f'[{last}][clip{i}]overlay={x}:{y}:eof_action=repeat[out{i}]']
        last = f'out{i}'
    # Re-number the standalone cut and preserve Figure 2's source qualifier above y=1020.
    footer = (f'drawbox=x=0:y=1020:w=1920:h=60:color={BG}:t=fill,'
              f"drawtext=fontfile={FONT}:text='CAST / ICRA supplementary video':x=80:y=1032:fontsize=18:fontcolor={MUTED},"
              f"drawtext=fontfile={FONT}:text='{slide['id']} / {len(story['slides']):02}':x=1740:y=1032:fontsize=18:fontcolor={MUTED}")
    if slide['id'] == '01':
        footer += ',fade=t=in:st=0:d=0.3:color=white'
    if slide is story['slides'][-1]:
        footer += f",fade=t=out:st={slide['duration']-0.4}:d=0.4:color=white"
    filters.append(f'[{last}]{footer},setsar=1[final]')
    ffmpeg(*cmd, '-filter_complex_threads', '2', '-filter_complex', ';'.join(filters),
           '-map', '[final]', '-t', slide['duration'], '-an', '-r', FPS, '-c:v', 'libx264',
           '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '4',
           '-video_track_timescale', '15360', part)
    slide['segment'] = str(part.relative_to(OUT))
    slide['media_sources'] = [str((ROOT / m['filename']).relative_to(ROOT)) for m in active]
    print(f"Rendered scene {slide['id']}: {slide['section']} ({slide['duration']} s)", flush=True)
    return part


def finish_exports(parts, story):
    listing = OUT / 'segments' / 'concat.txt'
    listing.write_text(''.join(f"file '{p.name}'\n" for p in parts))
    stem = story.get('file_stem', 'cast_paper_explanation')
    master = OUT / f'{stem}.mp4'
    ffmpeg('-f', 'concat', '-safe', '0', '-i', listing, '-i', OUT / 'chapters.ffmetadata',
           '-i', OUT / 'captions.srt', '-map', '0:v:0', '-map', '2:s:0', '-map_metadata', '1',
           '-map_chapters', '1', '-c:v', 'copy', '-c:s', 'mov_text', '-disposition:s:0', '0',
           '-metadata:s:s:0', 'language=eng', '-metadata:s:s:0', 'title=English narration cues',
           '-movflags', '+faststart', master)
    print('Wrote editing master with optional English subtitle track.', flush=True)
    captioned = OUT / f'{stem}_captioned.mp4'
    # Dedicated 108-pixel caption band avoids covering equations or result labels.
    ffmpeg('-i', master, '-map', '0:v:0', '-vf',
           f"scale=1728:972:flags=lanczos,pad=1920:1080:96:0:color={BG},drawbox=x=0:y=972:w=1920:h=108:color=0xF5F6F7:t=fill,ass=filename={OUT/'captions.ass'}",
           '-an', '-sn', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
           '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart', captioned)
    print('Wrote captioned rehearsal version.', flush=True)
    return master, captioned


def write_review_files(story, videos):
    shots = [4, 15, 22, 32, 35, 41, 47, 52, 61, 70, 85, 99]
    sheet = Image.new('RGB', (1280, ((len(shots)+1)//2)*390), BG)
    font = ImageFont.truetype(FONT, 22)
    for i, t in enumerate(shots):
        dest = OUT / 'frames' / f'{t:03}.jpg'
        ffmpeg('-ss', t, '-i', videos[1], '-frames:v', '1', '-update', '1', dest)
        im = Image.open(dest); im.thumbnail((640, 360))
        x, y = i % 2 * 640, i // 2 * 390
        sheet.paste(im, (x, y))
        ImageDraw.Draw(sheet).text((x+14, y+363), clock(t), font=font, fill=MUTED)
    sheet.save(OUT / 'contact_sheet.jpg', quality=92)
    results = {}
    for path in videos:
        p = probe(path); v = next(s for s in p['streams'] if s['codec_type']=='video')
        assert (v['width'], v['height'], v['r_frame_rate']) == (1920,1080,'30/1')
        assert v['pix_fmt']=='yuv420p' and int(v['nb_frames'])==story['duration_seconds']*FPS
        assert abs(float(p['format']['duration']) - story['duration_seconds']) < 1/FPS
        assert not any(s['codec_type']=='audio' for s in p['streams'])
        assert len(p['chapters']) == len(story['slides'])
        run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(path),'-map','0:v:0','-f','null','-'])
        results[path.name] = dict(duration_seconds=float(p['format']['duration']),
                                 width=v['width'],height=v['height'],fps=v['r_frame_rate'],
                                 frames=int(v['nb_frames']),bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT / 'validation.json').write_text(json.dumps(results,indent=2)+'\n')
    readme = f'''# CAST — paper explanation video

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
'''
    (OUT / 'README.md').write_text(readme)
    (OUT / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST · paper explanation</title><style>:root{color-scheme:light;background:#ffffff;color:#202124;font-family:system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}video{width:100%;aspect-ratio:16/9;background:#ffffff}p{line-height:1.6;color:#626b73}a{color:#287764}button{font:inherit;background:#ffffff;color:#202124;border:1px solid #d5d9dd;border-radius:6px;padding:10px 14px;margin:0 10px 16px 0;cursor:pointer}</style><main><h1>CAST · paper explanation</h1><p>1:44 · 1080p · ready for your narration. The remaining 1:16 is available for simulation and real-robot footage.</p><button onclick="pick(true)">Captioned rehearsal</button><button onclick="pick(false)">Clean master</button><video id="v" controls playsinline preload="metadata" poster="frames/047.jpg" src="cast_paper_explanation_captioned.mp4"></video><p><a href="voiceover_script.md">Read the script</a> · <a href="voiceover_cue_sheet.md">Recording cues</a> · <a href="captions.srt">Subtitles</a> · <a href="README.md">Editing notes</a></p></main><script>function pick(captions){const v=document.getElementById('v');const t=v.currentTime;v.pause();v.src=captions?'cast_paper_explanation_captioned.mp4':'cast_paper_explanation.mp4';v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration)}};</script></html>''')


def main():
    for name in ['', 'assets', 'segments', 'frames']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    # Regenerate only when the academic animation sources or inputs have changed.
    animation_dir = OUT/'animations'
    sources = [ROOT/'scripts/build_animations.py', ROOT/'scripts/figure2_animation.py',
               ROOT/'assets/experiment_data.json', ROOT.parent/'_ICRA_2027__CAST/figures/method_update.npz',
               ROOT.parent/'_ICRA_2027__CAST/figures/method_update.json']
    latest=max(p.stat().st_mtime for p in sources)
    clips=[animation_dir/name for name in ['A01_accumulated_drift.mp4','A02_paired_residuals.mp4',
           'A03_cast_update.mp4','A04_regress_refresh.mp4','A05_measured_learning_progress.mp4']]
    if any(not p.exists() or p.stat().st_mtime < latest for p in clips):
        env=dict(os.environ, CAST_VIDEO_THEME='academic', CAST_VIDEO_ANIMATION_DIR=str(animation_dir))
        subprocess.run([sys.executable,str(ROOT/'scripts/build_animations.py')],env=env,check=True)
    build_academic_slides()
    story = prepare_story()
    story['style']='Academic: white canvas, dark text, restrained blue/orange/green, flat plots.'
    assert story['duration_seconds']==104
    assert [s['source_slide_id'] for s in story['slides']]==['01','02','03','04','05','06','10']
    slots = json.loads((ROOT / 'animation_manifest.json').read_text())['slots']
    write_documents(story)
    evidence = build_evidence_clip()
    print('Prepared the benchmark overview and unchanged measured chart.', flush=True)
    parts = [render_scene(s,story,slots,evidence) for s in story['slides']]
    # Store the exact source paths after rendering; no rollout clip may be referenced.
    for slide in story['slides']:
        assert all('footage/' not in p and 'V01' not in p and 'V02' not in p for p in slide['media_sources'])
    write_documents(story)
    videos = finish_exports(parts, story)
    write_review_files(story, videos)
    print(f'Validated both 104-second videos. Open {OUT / "index.html"}', flush=True)


if __name__=='__main__':
    main()
