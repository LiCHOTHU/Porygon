#!/usr/bin/env python3
"""Academic side-by-side comparisons from the supplied simulator recordings."""
from pathlib import Path
import hashlib,json
from academic_video_slides import Slide, INK, MUTED, GREEN
import build_explanation_video as export

ROOT=Path(__file__).resolve().parents[1]
SPECS={
 'sim_libero':dict(stem='libero_task32_learning',duration=11,frames=503,
     title='LIBERO: place the bottle in the drawer',
     subtitle='Frozen base and CAST · same initial state',policy='Frozen base versus CAST',
     crop=[512,512,0,64],boxes=[[160,285,700,700],[1060,285,700,700]],
     stages=[dict(label='Frozen base',start_frame=0,end_frame=315),
             dict(label='CAST',start_frame=315,end_frame=503)],
     cues=[(0,6,'Both policies begin from the same initial state.','Now let’s look at the experiments. In LIBERO, both policies start from the same state.'),
           (6,11,'The right-hand CAST rollout places the bottle in the drawer; the left-hand base fails.','CAST places the bottle in the drawer; the base fails.')]),
 'sim_robomimic':dict(stem='robomimic_square_learning',duration=10,frames=665,
     title='robomimic: learning square-nut insertion',
     subtitle='CAST policy · three training checkpoints',policy='CAST',
     crop=[512,512,0,64],boxes=[[70,335,560,560],[680,335,560,560],[1290,335,560,560]],
     stages=[dict(label='Early · step 2,000',start_frame=0,end_frame=414),
             dict(label='Mid · step 16,000',start_frame=414,end_frame=541),
             dict(label='Final · step 31,000',start_frame=541,end_frame=665)],
     cues=[(0,4,'Compare the three supplied CAST checkpoints side by side.','To see how learning progresses, robomimic compares three CAST checkpoints.'),
           (4,10,'The middle and final policies place the nut onto the square peg; their last frames remain visible.','The early policy struggles, while later checkpoints complete the nut insertion.')]),
 'sim_dmc':dict(stem='dmc_walker_run_learning',duration=11,frames=762,
     title='DMC walker-run: locomotion learning',
     subtitle='CAST policy · three training checkpoints',policy='CAST',
     crop=[512,384,0,128],boxes=[[70,355,560,420],[680,355,560,420],[1290,355,560,420]],
     stages=[dict(label='Early',start_frame=0,end_frame=254),
             dict(label='Mid-training',start_frame=254,end_frame=508),
             dict(label='Final',start_frame=508,end_frame=762)],
     cues=[(0,11,'CAST training checkpoints show a walker progressing toward running.','Beyond manipulation, DMC tests locomotion with dense rewards. As CAST learns, the walker progresses from unsteady movement toward running.')]),
}


def prepare(out):
    provenance={};backgrounds={}
    for key,spec in SPECS.items():
        source=ROOT/'task_demos'/f"{spec['stem']}.mp4"
        sidecar=source.with_suffix('.json')
        info=export.probe(source);v=next(s for s in info['streams'] if s['codec_type']=='video')
        assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(512,576,'20/1',spec['frames'])
        record=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    metadata=str(sidecar),metadata_contents=json.loads(sidecar.read_text()),
                    actual_source_fps=20,actual_source_frames=spec['frames'],
                    policy=spec['policy'],
                    policy_attribution_basis='Author confirmed that the supplied successful demonstrations use CAST-trained policies. Original source metadata is retained verbatim.',
                    output_duration_seconds=spec['duration'],
                    image_crop_width_height_x_y=spec['crop'],
                    crop_note='Remove the original title banner and, for DMC, black letterbox bars; retain the complete simulator image.',
                    layout='Complete stage intervals shown simultaneously. Shorter stages play at source speed and hold their last frame; longer stages accelerate to fit.',
                    boundary_detection='Stage title changes in the actual encoded video; timing uses ffprobe rather than stale duration fields in the sidecars.',stages=[])
        slide=Slide(spec['title']);slide.text(80,163,spec['subtitle'],28,MUTED,width=1760)
        for stage,box in zip(spec['stages'],spec['boxes']):
            count=stage['end_frame']-stage['start_frame'];speed=max(1.0,count/20/spec['duration'])
            motion_duration=count/20/speed
            assert speed>=1
            record['stages'].append(dict(stage,source_start_seconds=stage['start_frame']/20,
                                         source_end_seconds=stage['end_frame']/20,source_frame_count=count,
                                         speed_relative_to_source=speed,output_motion_duration_seconds=motion_duration,
                                         last_frame_hold_seconds=spec['duration']-motion_duration,output_box=box))
            x,y,w,h=box
            slide.text(x,y-92,stage['label'],32,GREEN if stage['label']=='CAST' else INK,bold=True,width=w)
            slide.text(x,y-47,f'{speed:.2f}× source playback',23,MUTED,width=w)
        if key=='sim_dmc':slide.text(80,885,'Continuous control with dense rewards',29,MUTED,width=1760)
        if key=='sim_robomimic':slide.text(80,948,'Individual evaluation rollouts from the three checkpoints',26,MUTED,width=1760)
        path=out/'assets'/f'{key}.png';slide.im.save(path);backgrounds[key]=path;provenance[key]=record
    (out/'simulation_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance,backgrounds


def story_scenes(factory):
    return [factory(spec['title'],spec['title'],spec['duration'],key,spec['cues']) for key,spec in SPECS.items()]


def append_filters(s,record,cmd,filters,last):
    spec=SPECS[s['kind']]
    cmd+=['-i',record['source']]
    n=len(record['stages']);filters.append(f"[1:v]split={n}"+''.join(f'[source{i}]' for i in range(n)))
    w,h,cx,cy=spec['crop']
    for i,stage in enumerate(record['stages']):
        x,y,ow,oh=stage['output_box'];speed=stage['speed_relative_to_source']
        filters += [f"[source{i}]trim=start_frame={stage['start_frame']}:end_frame={stage['end_frame']},"
                    f"setpts=(PTS-STARTPTS)/{speed:.10f},crop={w}:{h}:{cx}:{cy},fps=30,"
                    f"scale={ow}:{oh}:flags=lanczos,setsar=1[stage{i}]",
                    f'[{last}][stage{i}]overlay={x}:{y}:eof_action=repeat[sim{i}]']
        last=f'sim{i}'
    return last
