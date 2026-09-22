#!/usr/bin/env python3
"""Export the timed slides + MP4 overlays as a silent 1080p video preview."""
from pathlib import Path
import argparse,json,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true',help='Include the final minute, which still has robot-footage placeholders.');args=p.parse_args()
    story=json.loads((ROOT/'storyboard.json').read_text())
    slots=json.loads((ROOT/'animation_manifest.json').read_text())['slots']
    selected=story['slides'] if args.full else story['slides'][:6]
    output=ROOT/('cast_full_silent_draft.mp4' if args.full else 'cast_method_preview.mp4')
    with tempfile.TemporaryDirectory(prefix='cast-video-') as directory:
        temp=Path(directory);parts=[]
        for s in selected:
            sid=int(s['id']);part=temp/f'{sid:02d}.mp4';parts.append(part)
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','30','-i',str(ROOT/'previews'/f'slide-{sid:02d}.png')]
            active=[m for m in slots if m['slide']==sid and m['status'].startswith('completed')]
            for m in active:cmd+=['-i',str(ROOT/m['filename'])]
            filters=[];last='0:v'
            for i,m in enumerate(active,1):
                x,y,w,h=[round(2*v) for v in m['playback_box_points']]
                # yuv420p output needs even dimensions; rounding does not stretch appreciably.
                w=w//2*2;h=h//2*2
                filters += [f'[{i}:v]scale={w}:{h},setsar=1[clip{i}]',f'[{last}][clip{i}]overlay={x}:{y}:eof_action=repeat[out{i}]']
                last=f'out{i}'
            if filters:cmd+=['-filter_complex',';'.join(filters),'-map',f'[{last}]']
            cmd+=['-t',str(s['duration']),'-an','-r','30','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-threads','4',str(part)]
            subprocess.run(cmd,check=True)
            print(f'Rendered slide {sid:02d}: {s["duration"]} seconds',flush=True)
        listing=temp/'concat.txt';listing.write_text(''.join(f"file '{part}'\n" for part in parts))
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(output)],check=True)
    print(f'Wrote {output}',flush=True)
if __name__=='__main__':main()
