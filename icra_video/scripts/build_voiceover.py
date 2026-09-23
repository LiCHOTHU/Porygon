#!/usr/bin/env python3
"""Export the spoken script and timed recording cues from the video storyboard."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def timecode(seconds):
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def cue_notes(slide):
    """Absolute edit times for PowerPoint's presenter notes."""
    return '\n\n'.join(
        f"{timecode(slide['start'] + cue['start'])}–"
        f"{timecode(slide['start'] + cue['end'])}\n"
        f"ON SCREEN: {cue['visual']}\nSAY: {cue['text']}"
        for cue in slide['voiceover_cues']
    )


def write_voiceover_documents(story):
    # Keep the root recording copy aligned to the final video when rebuilding
    # the earlier slide deck. Its original slide notes still use cue_notes().
    final = ROOT / 'supplementary'
    if all((final / name).exists() for name in ['edit_manifest.json', 'voiceover_script.md', 'voiceover_cue_sheet.md']):
        (ROOT / 'voiceover_script.md').write_text((final / 'voiceover_script.md').read_text())
        cue_text = (final / 'voiceover_cue_sheet.md').read_text().replace('(timeline.md)', '(supplementary/timeline.md)')
        (ROOT / 'voiceover_cue_sheet.md').write_text(cue_text + '\nVideo: [captioned cut](supplementary/cast_supplementary_video_captioned.mp4). '
                                                  'This edit is defined in `supplementary/edit_manifest.json`; CSV timing files are in `supplementary/`.\n')
        return
    total = story['total_duration_seconds']
    reserve = story['maximum_duration_seconds'] - total
    words = sum(len(s['narration'].split()) for s in story['slides'])
    # This copy contains only spoken paragraphs, one per slide, for direct reading.
    script = ['# CAST — spoken script', '']
    cue_sheet = [
        '# CAST — timed voiceover cue sheet', '',
        '**Production notes; only the “Say” column is spoken.** '
        'All timestamps are absolute positions in the current 2:36 edit. '
        'Cue boundaries indicate when to begin each phrase; allow natural pauses within the interval. '
        'The clean reading copy is [voiceover_script.md](voiceover_script.md); '
        'its ten paragraphs follow the ten slides in order.', '',
        f'**Current cut: {timecode(total)} · approximately {words} words.** '
        f'{reserve} seconds remain within the three-minute limit for incoming simulation footage. '
        'These are delivery targets, not recorded speech durations. '
        'Rehearse against [the preview](preview.html), allowing natural pauses. '
        'Say CAST as “cast,” DICE-RL as “dice R L,” and DMC as “D M C.”', '',
        'The simulation montage currently shows labeled demonstrations and a scripted DMC controller. '
        'The robot slots currently show labeled demonstration stills. '
        'Narration reports the paper’s measured results without attributing these illustrative images to a CAST rollout.', '',
    ]
    for slide in story['slides']:
        cues = slide['voiceover_cues']
        assert ' '.join(c['text'] for c in cues) == slide['narration']
        assert cues[0]['start'] == 0 and cues[-1]['end'] == slide['duration']
        end = 0
        for cue in cues:
            assert cue['start'] == end < cue['end'] <= slide['duration']
            end = cue['end']
        heading = (f"## Slide {slide['id']} · {timecode(slide['start'])}–"
                   f"{timecode(slide['start'] + slide['duration'])} · {slide['section']}")
        script += [slide['narration'], '']
        cue_sheet += [heading, '', '| Time | On screen / delivery cue | Say |',
                      '|---|---|---|']
        for cue in cues:
            start, end = slide['start'] + cue['start'], slide['start'] + cue['end']
            cue_sheet.append(f"| {timecode(start)}–{timecode(end)} | {cue['visual']} | {cue['text']} |")
        cue_sheet.append('')
    cue_sheet += [
        '## Recording and editing notes', '',
        '- Figure 2 occupies 00:26–00:56. The critic request appears at 00:31 and contracts at 00:34. '
        'The combined update contracts around the **current action** at 00:44–00:50. '
        'Keep those two clipping operations distinct in the delivery.',
        '- Slide 5 shows a toy fitting process. Restoration is a soft pull; the circle on slide 4 bounds the target step. '
        'Neither visual asserts a hard bound on the fitted policy’s output.',
        '- Slide 6 switches from environment footage to the measured LIBERO chart at 01:15. '
        'The robomimic efficiency chart is visible throughout. The 38% comparison is against DICE-RL '
        'at the first evaluation reaching 90% success.',
        '- The LIBERO bars aggregate five tasks. The spoken comparison with radial-only uses the underlying '
        'per-task results: CAST improves every task over the base; radial-only exceeds CAST on two.',
        '- Slides 8 and 9 read the base-to-CAST change. The other percentages remain visible on screen; '
        'leave room for the task explanation instead of reading every bar.',
        '- When simulation footage is inserted, add its narration here and shift all subsequent cue times. '
        'The present script follows the existing edit exactly; it does not include an unfilled 24-second speaking segment.', '',
        '## Updating the draft', '',
        '`storyboard.json` is the source for spoken text and relative cue times. '
        'Keep each slide’s `narration` equal to its cue texts joined with spaces. '
        'Rebuild the deck and preview to update their notes. '
        'Both reading documents are generated by `scripts/build_voiceover.py`, which the slide builder also calls.', '',
    ]
    (ROOT / 'voiceover_script.md').write_text('\n'.join(script))
    (ROOT / 'voiceover_cue_sheet.md').write_text('\n'.join(cue_sheet))


if __name__ == '__main__':
    write_voiceover_documents(json.loads((ROOT / 'storyboard.json').read_text()))
    print('Wrote the spoken script and timed voiceover cue sheet.')
