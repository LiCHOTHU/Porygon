#!/usr/bin/env python3
"""Assemble the CAST explanation, simulator learning clips, and robot examples."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont
from academic_video_slides import Slide, FONT, INK, MUTED, GREEN, BLUE, LINE
import build_explanation_video as export
import simulation_video_scenes as simulations
import supplementary_delivery

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'supplementary'
RECORDINGS=Path('/home/licho/workspace/real_policy_eval')
SOURCES={
    'placement':dict(take='jigglypuff/20260913_004154_978275_4b21de',
                     attribution_meta='jigglypuff/20260914_181014_490638_b0f486/take_meta.json',
                     title='Jigglypuff placement',trim_start=0,duration=11),
    'stacking':dict(take='stack/20260914_005719_393758_e64e62',
                    attribution_meta='stack/20260914_005719_393758_e64e62/take_meta.json',
                    title='Red-on-purple stacking',trim_start=0,duration=11),
}
CAMERAS=[('chest','Chest camera',(80,240,864,648)),
         ('left_wrist','Wrist camera',(976,240,864,648))]


def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sources():
    provenance={}
    for key,item in SOURCES.items():
        take=RECORDINGS/item['take'];meta_path=take/'take_meta.json'
        meta=json.loads(meta_path.read_text());ev=meta['evaluation'];checkpoint=ev['checkpoint']
        assert ev['success'] is True and ev['outcome']=='success'
        assert ev['review_status']=='saved' and not ev.get('recording_errors')
        attribution_path=RECORDINGS/item['attribution_meta']
        attribution=json.loads(attribution_path.read_text())['evaluation']['checkpoint']
        assert attribution['path']==checkpoint['path']
        record=dict(task=key,source_take=str(take),metadata=str(meta_path),
                    metadata_sha256=sha256(meta_path),success_label=True,
                    checkpoint=checkpoint['path'],policy_label='CAST',
                    policy_attribution_basis='Author confirmed these successful robot recordings use CAST-trained policies. Recorded checkpoint identifiers are retained as source metadata.',
                    attribution_metadata=str(attribution_path),
                    recorded_checkpoint_display_name=attribution['display_name'],
                    source_start_seconds=item['trim_start'],source_duration_seconds=item['duration'],
                    playback_speed=1.0,camera_alignment='Shared take; both camera tracks start at frame zero and have 15 fps.',
                    visual_review=('Toy released fully inside the case, visible in both views.' if key=='placement' else
                                   'Red cube rests on purple cube; wrist view shows fingers open and clear of the red cube.'),
                    streams={})
        for camera,_,_ in CAMERAS:
            path=next(take.glob(camera+'_rgb*.avi'))
            info=export.probe(path);v=next(s for s in info['streams'] if s['codec_type']=='video')
            assert float(v['duration'])>=item['trim_start']+item['duration']
            assert v['r_frame_rate']=='15/1'
            record['streams'][camera]=dict(path=str(path),sha256=sha256(path),frames=int(v['nb_frames']),
                                          duration_seconds=float(v['duration']),fps=v['r_frame_rate'],
                                          width=v['width'],height=v['height'])
            still=OUT/'assets'/f'{key}_{camera}_start.png'
            export.ffmpeg('-ss','0.1','-i',path,'-frames:v','1','-update','1',still)
        provenance[key]=record
    (ROOT/'robot_rollouts/selection.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUT/'rollout_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def scene(section,title,duration,kind,cues):
    voice=[dict(start=a,end=b,visual=v,text=t) for a,b,v,t in cues]
    return dict(section=section,title=title,duration=duration,kind=kind,voiceover_cues=voice,
                narration=' '.join(c['text'] for c in voice))


def build_story():
    paper=json.loads((ROOT/'explanation/edit_manifest.json').read_text())
    scenes=copy.deepcopy(paper['slides'][:5])
    for s in scenes:
        s['kind']='existing'
        s['source_segment']=str(ROOT/'explanation'/s['segment'])
    scenes += simulations.story_scenes(scene)
    evidence=copy.deepcopy(paper['slides'][5])
    evidence['kind']='existing';evidence['source_segment']=str(ROOT/'explanation'/evidence['segment'])
    evidence['source_trim_start']=9;evidence['duration']=21
    # The author shortened this discussion in the reading draft. Preserve that
    # revision while connecting the demonstrations to the aggregate results.
    evidence['voiceover_cues']=[
        dict(start=0,end=3,
             visual='Transition from the task demonstrations to the measured simulation results.',
             text='The measured results show the same pattern.'),
        dict(start=3,end=12,
             visual='Read the right-hand robomimic square chart: steps to the first evaluation reaching 90% success.',
             text='In robomimic square, CAST reaches ninety percent success using thirty-eight percent fewer environment steps than DICE-RL.'),
        dict(start=12,end=17,
             visual='The left-hand LIBERO chart summarizes five tasks; no controls and clipping alone yield zero success.',
             text='Across five LIBERO tasks, no controls or clipping alone gives zero success,'),
        dict(start=17,end=21,
             visual='CAST improves over the base on every task; the displayed bars aggregate the five tasks.',
             text='while CAST improves every task over the base.')]
    evidence['narration']=' '.join(c['text'] for c in evidence['voiceover_cues'])
    scenes.append(evidence)
    scenes += [
        scene('Real robot setup','From offline refinement to robot execution',16,'setup',[
            (0,5,'Chest and wrist views from a recorded robot evaluation.',
             'We next move from simulation to OpenArm, using chest and wrist cameras.'),
            (5,11,'Offline CAST refinement uses recorded successful and failed rollouts.',
             'CAST learns offline from recorded successes and failures before deployment.'),
            (11,16,'Introduce the successful CAST placement and stacking rollouts.',
             'We now show the CAST policies completing placement and stacking.')]),
        scene('Successful placement example','Jigglypuff placement',11,'placement',[
            (0,5,'The robot approaches and grasps the toy on a stair.',
             'For placement, the robot reaches down and grasps Jigglypuff.'),
            (5,8,'The gripper transports the toy over the case.',
             'It carries the toy over the case,'),
            (8,11,'The fingers open; the toy falls fully inside the case.',
             'then releases it fully inside.')]),
        scene('Successful stacking example','Red-on-purple stacking',11,'stacking',[
            (0,4,'The gripper approaches and grasps the red cube.',
             'For stacking, it grasps the red cube,'),
            (4,8,'The held red cube moves above the purple support cube.',
             'then aligns it above the purple cube.'),
            (8,11,'Wrist view shows release and the completed stack.',
             'The gripper opens, completing the stack.')]),
        scene('Reported robot results','Reported robot success rates',14,'results',[
            (0,7,'Placement chart from the paper: base 55%, CAST 95%.',
             'Looking at the overall results, CAST raises placement success from 55 to 95 percent.'),
            (7,14,'Stacking chart from the paper: base 35%, CAST 70%.',
             'Stacking improves from 35 to 70 percent, again exceeding both refinement baselines.')])]
    closing=copy.deepcopy(paper['slides'][-1]);closing['kind']='existing'
    closing['source_segment']=str(ROOT/'explanation'/closing['segment']);scenes.append(closing)
    elapsed=0
    for i,s in enumerate(scenes,1):
        s['id']=f'{i:02}';s['start']=elapsed;elapsed+=s['duration']
        for cue in s['voiceover_cues']:
            spoken=cue['text'].replace('DICE-RL','dice R L').replace('DMC','D M C')
            words=len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",spoken))
            assert words*60/(cue['end']-cue['start'])<=160,(s['id'],cue['text'])
    assert elapsed==179
    return dict(title='CAST — supplementary video',file_stem='cast_supplementary_video',
                duration_seconds=elapsed,maximum_final_duration_seconds=180,
                remaining_footage_budget_seconds=180-elapsed,slides=scenes,
                style=paper['style'],audio='No recorded narration yet; ready for the author’s voiceover.',
                footage='CAST learning in three simulators and two successful CAST robot rollouts at 1× speed, followed by aggregate evaluation results. Policy attribution follows the author’s clarification.',
                sources=['../explanation/edit_manifest.json','simulation_sources.json','rollout_sources.json','../assets/robot_results.json'])


def build_robot_backgrounds():
    paths={}
    s=Slide('Real robot evaluation')
    for camera,label,box in CAMERAS:
        s.text(box[0],185,label,32,bold=True)
        s.image(OUT/'assets'/f'placement_{camera}_start.png',box)
    s.text(80,934,'Recorded rollouts  →  Offline CAST refinement  →  Autonomous evaluation',29,MUTED,width=1760)
    paths['setup']=OUT/'assets/setup.png';s.im.save(paths['setup'])
    for key,item in SOURCES.items():
        s=Slide(item['title'])
        s.text(80,158,'Successful CAST rollout · 1× speed',26,MUTED)
        for _,label,(x,y,w,h) in CAMERAS:s.text(x,201,label,27,bold=True)
        caption=('Grasp the toy → transport to the case → release inside' if key=='placement' else
                 'Grasp the red cube → align above the purple cube → release')
        s.text(80,934,caption,29,MUTED,width=1760)
        paths[key]=OUT/'assets'/f'{key}.png';s.im.save(paths[key])
    s=Slide('Reported robot success rates')
    data=json.loads((ROOT/'assets/robot_results.json').read_text())
    for origin,key,title in [(80,'jigglypuff_into_case','Placement'),(1000,'red_on_purple_stacking','Stacking')]:
        s.text(origin,204,title,38,bold=True)
        bx=origin+250;barw=445
        rows=data['tasks'][key]['results']
        for i,(label,row) in enumerate(zip(['Base','BC refinement','DICE-RL','CAST'],rows)):
            pct=100*row['successes']/row['trials'];y=325+i*108
            color=GREEN if label=='CAST' else BLUE if label=='DICE-RL' else '#8A9299'
            s.text(origin,y+12,label,29,color,bold=label=='CAST',width=238)
            end=bx+barw*pct/100
            s.d.rectangle((bx,y,end,y+52),fill=color)
            s.text(end+13,y+10,f'{pct:.0f}%',28,bold=label=='CAST')
        y=775;s.d.line((bx,y,bx+barw,y),fill=LINE,width=2)
        for pct in [0,25,50,75,100]:
            x=bx+barw*pct/100;s.d.line((x,y,x,y+8),fill=MUTED,width=2);s.text(x-12,y+14,str(pct),22,MUTED)
        s.text(bx+105,842,'Success rate (%)',27,MUTED)
    s.text(80,942,'Aggregate results from the paper; the preceding clips illustrate the tasks.',28,MUTED,width=1760)
    paths['results']=OUT/'assets/results.png';s.im.save(paths['results'])
    return paths


def render_scene(s,story,backgrounds,provenance,sim_provenance):
    part=OUT/'segments'/f"{s['id']}_{s['kind']}.mp4"
    if s['kind']=='existing':cmd=['-ss',s.get('source_trim_start',0),'-i',s['source_segment']]
    else:cmd=['-loop','1','-framerate','30','-i',backgrounds[s['kind']]]
    filters=[];last='0:v'
    if s['kind'] in simulations.SPECS:
        last=simulations.append_filters(s,sim_provenance[s['kind']],cmd,filters,last)
    if s['kind'] in SOURCES:
        source=provenance[s['kind']]
        for i,(camera,_,(x,y,w,h)) in enumerate(CAMERAS,1):
            cmd+=['-ss',source['source_start_seconds'],'-i',source['streams'][camera]['path']]
            filters += [f'[{i}:v]trim=duration={s["duration"]},setpts=PTS-STARTPTS,fps=30,scale={w}:{h}:flags=lanczos,setsar=1[c{i}]',
                        f'[{last}][c{i}]overlay={x}:{y}:shortest=1[o{i}]']
            last=f'o{i}'
    footer=(f'drawbox=x=0:y=1020:w=1920:h=60:color=white:t=fill,'
            f"drawtext=fontfile={FONT}:text='CAST / ICRA supplementary video':x=80:y=1032:fontsize=18:fontcolor={MUTED},"
            f"drawtext=fontfile={FONT}:text='{s['id']} / {len(story['slides']):02}':x=1740:y=1032:fontsize=18:fontcolor={MUTED}")
    filters.append(f'[{last}]{footer},setsar=1[out]')
    export.ffmpeg(*cmd,'-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]',
                  '-t',s['duration'],'-an','-r','30','-c:v','libx264','-preset','fast','-crf','18',
                  '-pix_fmt','yuv420p','-threads','4','-video_track_timescale','15360',part)
    s['segment']=str(part.relative_to(OUT))
    if s['kind'] in SOURCES:s['rollout_source']=provenance[s['kind']]
    if s['kind'] in simulations.SPECS:s['simulation_source']=sim_provenance[s['kind']]
    print(f"Rendered {s['id']}: {s['section']}",flush=True)
    return part


def review_and_validate(story,videos):
    by_kind={s['kind']:s for s in story['slides']}
    times=[47]
    for key in ['sim_libero','sim_robomimic','sim_dmc']:
        s=by_kind[key];times += [s['start']+s['duration']//2,s['start']+s['duration']-1]
    times += [story['slides'][8]['start']+13,by_kind['setup']['start']+6]
    for key in ['placement','stacking']:
        s=by_kind[key];times += [s['start']+5,s['start']+s['duration']-1]
    times += [by_kind['results']['start']+8,story['slides'][-1]['start']+4]
    sheet=Image.new('RGB',(1280,((len(times)+1)//2)*385),'white');f=ImageFont.truetype(FONT,20)
    for i,t in enumerate(times):
        p=OUT/'frames'/f'{t:03}.jpg'
        export.ffmpeg('-ss',t,'-i',videos[1],'-frames:v','1','-update','1',p)
        im=Image.open(p);im.thumbnail((640,360));x=i%2*640;y=i//2*385
        sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+8,y+360),export.clock(t),font=f,fill=MUTED)
    sheet.save(OUT/'contact_sheet.jpg',quality=82)
    validation={}
    for p in videos:
        info=export.probe(p);v=next(x for x in info['streams'] if x['codec_type']=='video')
        assert (v['width'],v['height'],v['r_frame_rate'],v['pix_fmt'])==(1920,1080,'30/1','yuv420p')
        assert int(v['nb_frames'])==story['duration_seconds']*30
        assert abs(float(info['format']['duration'])-story['duration_seconds'])<1/30
        assert len(info['chapters'])==len(story['slides'])
        assert not any(s['codec_type']=='audio' for s in info['streams'])
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(p),'-map','0:v:0','-f','null','-'],check=True)
        validation[p.name]=dict(duration_seconds=float(info['format']['duration']),frames=int(v['nb_frames']),
                                 resolution=[1920,1080],fps=30,bytes=p.stat().st_size,sha256=sha256(p),decode='passed')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')


def write_delivery_notes(story):
    supplementary_delivery.write(story,OUT)


def main():
    for name in ['assets','segments','frames']:(OUT/name).mkdir(parents=True,exist_ok=True)
    provenance=prepare_sources()
    story=build_story();backgrounds=build_robot_backgrounds()
    sim_provenance,sim_backgrounds=simulations.prepare(OUT)
    backgrounds.update(sim_backgrounds)
    export.OUT=OUT
    export.write_documents(story)
    parts=[render_scene(s,story,backgrounds,provenance,sim_provenance) for s in story['slides']]
    export.write_documents(story)
    videos=export.finish_exports(parts,story)
    review_and_validate(story,videos)
    write_delivery_notes(story)
    print(f'Completed and validated {OUT}/cast_supplementary_video_captioned.mp4',flush=True)


if __name__=='__main__':main()
