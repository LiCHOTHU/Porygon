#!/usr/bin/env python3
"""Package task demonstrations with labels and a nine-second montage."""
from pathlib import Path
import json,subprocess,tempfile,shutil
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'footage'
BG='#0D1827';WHITE='#F3F6FA';MUTED='#AFBDD0';TEAL='#42D7B4'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ITEMS=[
 dict(id='libero',title='LIBERO',task='Red mug → left plate',source='libero_demonstration.mp4',label='Dataset demonstration',caption='Image-based manipulation · sparse task-success reward',start=3.3333333333333335),
 dict(id='robomimic',title='robomimic',task='Square nut assembly',source='robomimic_demonstration.mp4',label='Dataset demonstration',caption='State-based manipulation · sparse task-success reward',start=3.3333333333333335),
 dict(id='dmc',title='DMC',task='Cartpole balance',source='dmc_cartpole_demonstration.mp4',label='Scripted controller',caption='Continuous control · dense reward',start=0),
]
def run(cmd):subprocess.run(cmd,check=True)
def probe(path):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
def main():
    OUT.mkdir(exist_ok=True);(OUT/'posters').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cast-montage-') as directory:
        temp=Path(directory);parts=[]
        for item in ITEMS:
            ident=item['id'];source=OUT/'source'/item['source'];assert source.exists()
            card=Image.new('RGB',(1280,720),BG);d=ImageDraw.Draw(card)
            title_font=ImageFont.truetype(BOLD,29);body_font=ImageFont.truetype(FONT,25)
            d.text((32,14),item['title'],font=title_font,fill=TEAL)
            advance=d.textlength(item['title'],font=title_font)
            d.text((53+advance,17),item['task'],font=body_font,fill=WHITE)
            d.text((1248,20),item['label'],font=ImageFont.truetype(FONT,20),fill=MUTED,anchor='ra')
            d.text((640,700),item['caption'],font=ImageFont.truetype(FONT,22),fill=MUTED,anchor='mm')
            background=temp/f'{ident}.png';card.save(background)
            target=OUT/f'{ident}_task_demo.mp4'
            run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','30','-i',str(background),'-i',str(source),'-filter_complex','[1:v]scale=1088:612,fps=30,setsar=1[clip];[0:v][clip]overlay=96:64:shortest=1[out]','-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)])
            poster=OUT/'posters'/f'{ident}.jpg'
            run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','3','-i',str(target),'-frames:v','1','-update','1',str(poster)])
            part=temp/f'{ident}.mp4';parts.append(part)
            run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(target),'-vf',f"trim=start_frame={round(item['start']*30)}:end_frame={round(item['start']*30)+90},setpts=PTS-STARTPTS",'-frames:v','90','-an','-r','30','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(part)])
            item.update(filename=target.name,poster=str(poster.relative_to(OUT)),duration_seconds=float(probe(target)['format']['duration']),montage_duration_seconds=3,playback_speed=1.0)
            print('Packaged',ident,flush=True)
        listing=temp/'parts.txt';listing.write_text(''.join(f"file '{p}'\n" for p in parts))
        montage=OUT/'benchmark_task_montage.mp4'
        run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(montage)])
        meta=probe(montage);assert abs(float(meta['format']['duration'])-9)<.04
        run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','1.5','-i',str(montage),'-frames:v','1','-update','1',str(OUT/'posters/montage.jpg')])
    # Keep the existing evidence slide at 30 seconds: 9 seconds of task context,
    # then 21 seconds of the original measured ablation animation.
    ablation=ROOT/'animations/A05_measured_learning_progress.mp4'
    if ablation.exists():
        composite=OUT/'A05_environments_and_results.mp4'
        run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(OUT/'benchmark_task_montage.mp4'),'-i',str(ablation),'-filter_complex','[0:v]trim=duration=9,setpts=PTS-STARTPTS[v0];[1:v]trim=duration=21,setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[out]','-map','[out]','-an','-r','30','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(composite)])
        shutil.copy2(ROOT/'animations/posters/A05_measured_learning_progress.png',OUT/'posters/A05_environments_and_results.png')
    record={'description':'Environment task illustrations: human demonstration state replays and a scripted DMC controller. These are not CAST policy evaluations.','montage':'benchmark_task_montage.mp4','duration_seconds':9,'clips':ITEMS}
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    gallery=[dict(title='Three benchmark tasks',filename=record['montage'],poster='posters/montage.jpg',label='9-second montage · all excerpts at normal speed')]+[dict(title=f"{i['title']} · {i['task']}",filename=i['filename'],poster=i['poster'],label=i['label']) for i in ITEMS]
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAST · benchmark task demonstrations</title><style>:root{color-scheme:dark;background:#0d1827;color:#f3f6fa;font-family:system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:30px}p{color:#afbdd0;line-height:1.6}nav{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}button{font:inherit;background:#16273b;color:#afbdd0;border:1px solid #33455b;border-radius:7px;padding:11px 15px;cursor:pointer}button.active{color:#42d7b4;border-color:#42d7b4}video{width:100%;aspect-ratio:16/9;background:#16273b}a{color:#42d7b4}</style><main><h1 id="title">Benchmark task demonstrations</h1><p id="label"></p><nav id="nav"></nav><video id="video" controls playsinline preload="metadata"></video><p>These clips illustrate the environments. LIBERO and robomimic replay recorded human demonstrations; DMC uses a scripted balancing controller.</p><p><a href="../preview.html">Animated slides</a> · <a href="README.md">Sources and reproduction</a></p></main><script>const clips=DATA;const v=document.getElementById('video');function pick(i,play=false){const c=clips[i];v.src=c.filename;v.poster=c.poster;document.getElementById('title').textContent=c.title;document.getElementById('label').textContent=c.label;document.querySelectorAll('button').forEach((b,j)=>b.classList.toggle('active',j===i));if(play)v.play().catch(()=>{})}clips.forEach((c,i)=>{const b=document.createElement('button');b.textContent=i?c.title.split(' · ')[0]:'Montage';b.onclick=()=>pick(i,true);document.getElementById('nav').append(b)});pick(0);</script></html>'''.replace('DATA',json.dumps(gallery))
    (OUT/'index.html').write_text(html)
    sheet=Image.new('RGB',(1280,2160),BG)
    for i,item in enumerate(ITEMS):sheet.paste(Image.open(OUT/item['poster']),(0,720*i))
    sheet.save(OUT/'contact_sheet.jpg',quality=92)
    print('Wrote nine-second montage and footage player.',flush=True)
if __name__=='__main__':main()
