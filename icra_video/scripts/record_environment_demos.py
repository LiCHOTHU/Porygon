#!/usr/bin/env python3
"""Render recorded human demonstration states, or a scripted DMC controller.

These clips illustrate benchmark tasks. They are not CAST evaluation rollouts.
Use liberocag for manipulation; river + /tmp/icra-video-dmc for DMC.
"""
import argparse, hashlib, importlib.util, json, os, subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'footage/source'
LIBERO_DATA=Path('/home/licho/workspace/imitation/data/libero/libero_90_unprocessed/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5')
SQUARE_DATA=Path('/tmp/icra-video-raw/square_ph_low_dim.hdf5')

class Writer:
    def __init__(self,path,width,height,fps):
        self.path=path;self.count=0
        self.proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,frame):self.proc.stdin.write(np.ascontiguousarray(frame).tobytes());self.count+=1
    def close(self):self.proc.stdin.close();assert self.proc.wait()==0

def relocate_assets(xml):
    root=ET.fromstring(xml)
    suite=Path(importlib.util.find_spec('robosuite').origin).parent
    libero_spec=importlib.util.find_spec('libero')
    libero=Path(libero_spec.origin).parent/'libero' if libero_spec else None
    changed=0
    for el in root.iter():
        old=el.get('file')
        if not old:continue
        new=old
        if '/robosuite/' in old:new=str(suite/'models/assets'/old.split('/models/assets/',1)[1])
        elif '/assets/' in old:
            relative=old.split('/assets/',1)[1]
            roots=[libero/'assets',Path(os.environ.get('ICRA_LIBERO_ASSETS','/home/licho/workspace/LIBERO/libero/libero/assets'))]
            candidates=[root/relative for root in roots]
            new=str(next((p for p in candidates if p.exists()),candidates[0]))
        if not Path(new).exists():raise FileNotFoundError(new)
        el.set('file',new);changed+=old!=new
    return ET.tostring(root,encoding='unicode'),changed

def replay(kind,path,demo='demo_0',limit=None):
    import h5py,mujoco
    from PIL import Image
    with h5py.File(path,'r') as f:
        group=f['data'][demo];states=group['states'][:]
        xml,relocations=relocate_assets(group.attrs['model_file'])
        rewards=group['rewards'][:] if 'rewards' in group else None
    # MuJoCo 3 removed legacy collision selection metadata; it does not change
    # replayed positions. Keep any compatibility edit explicitly recorded.
    tree=ET.fromstring(xml);compat=[]
    for opt in tree.findall('option'):
        if 'collision' in opt.attrib:compat.append('Removed obsolete option collision attribute');del opt.attrib['collision']
    xml=ET.tostring(tree,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    expected=1+model.nq+model.nv+model.na
    assert states.shape[1]==expected,(states.shape,expected,model.nq,model.nv,model.na)
    fps=20
    renderer=mujoco.Renderer(model,height=720,width=1280)
    visual=mujoco.MjvOption();visual.geomgroup[0]=0;visual.geomgroup[1]=1;visual.sitegroup[:]=0
    out=OUT/f'{kind}_demonstration.mp4';writer=Writer(out,1280,720,fps)
    for i,state in enumerate(states[:limit] if limit else states):
        data.time=state[0];data.qpos[:]=state[1:1+model.nq]
        data.qvel[:]=state[1+model.nq:1+model.nq+model.nv]
        if model.na:data.act[:]=state[-model.na:]
        mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera='agentview',scene_option=visual)
        image=renderer.render();writer.write(image)
        if i in [0,len(states)//2,len(states)-1]:Image.fromarray(image).save(OUT/f'{kind}_frame_{i:04d}.png')
        if i%40==0:print(kind,i,len(states),flush=True)
    writer.close();renderer.close()
    record={'benchmark':kind,'kind':'recorded human demonstration; simulator-state replay',
            'source_dataset':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'episode':demo,'frames':writer.count,'fps':fps,'duration_seconds':writer.count/fps,
            'camera':'agentview; widescreen aspect ratio, original pose and vertical field of view',
            'render_options':'visual meshes on; collision meshes and diagnostic sites hidden','resolution':[1280,720],'mujoco_version':mujoco.__version__,
            'asset_paths_relocated':relocations,'xml_compatibility_edits':compat,
            'states_replayed_exactly':True,'policy_checkpoint':None,'cast_evaluation':False}
    if rewards is not None:record['source_final_reward']=float(rewards[-1])
    (OUT/f'{kind}_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('WROTE',out,flush=True)

def dmc(seconds=12):
    from dm_control import suite
    from scipy.linalg import solve_discrete_are
    from PIL import Image
    import mujoco
    env=suite.load('cartpole','balance',task_kwargs={'random':285,'time_limit':20})
    env.reset();physics=env.physics;model=physics.model.ptr;data=physics.data.ptr
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    dt=env.control_timestep();substeps=round(dt/physics.timestep())
    def transition(x,u):
        mujoco.mj_resetData(model,data);data.qpos[:]=x[:2];data.qvel[:]=x[2:];data.ctrl[0]=u
        for _ in range(substeps):mujoco.mj_step(model,data)
        return np.r_[data.qpos,data.qvel].copy()
    eps=1e-5;zero=np.zeros(4)
    A=np.column_stack([(transition(np.eye(4)[i]*eps,0)-transition(-np.eye(4)[i]*eps,0))/(2*eps) for i in range(4)])
    B=((transition(zero,eps)-transition(zero,-eps))/(2*eps))[:,None]
    Q=np.diag([3.,35.,2.,4.]);R=np.array([[.3]])
    P=solve_discrete_are(A,B,Q,R);K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
    env.reset()
    with physics.reset_context():physics.data.qpos[:]=[0,.2];physics.data.qvel[:]=[0,0]
    writer=Writer(OUT/'dmc_cartpole_demonstration.mp4',1280,720,30)
    frames=int(seconds*30);controls=[];rewards=[];angles=[];next_frame=0
    for step in range(round(seconds/dt)):
        t=step*dt;x=np.r_[physics.data.qpos,physics.data.qvel]
        # A slow cart reference makes balancing motion visible. It is a scripted
        # control demonstration, independent of all paper checkpoints/results.
        reference=np.array([.55*np.sin(.85*t),0,.55*.85*np.cos(.85*t),0])
        u=float(np.clip((-K@(x-reference)).item(),-1,1))
        ts=env.step([u]);controls.append(u);rewards.append(float(ts.reward));angles.append(float(physics.data.qpos[1]))
        if step*dt+1e-8>=next_frame/30 and next_frame<frames:
            frame=physics.render(height=720,width=1280,camera_id=0);writer.write(frame)
            if next_frame in [0,180,359]:Image.fromarray(frame).save(OUT/f'dmc_frame_{next_frame:04d}.png')
            next_frame+=1
        if step%300==0:print('DMC seconds',t,flush=True)
    assert writer.count==frames,(writer.count,frames)
    writer.close()
    record={'benchmark':'DMC','task':'cartpole / balance','reward_setting':'dense',
            'kind':'scripted LQR controller in the actual DMC simulator','cast_evaluation':False,'policy_checkpoint':None,
            'seed':285,'initial_qpos':[0,.2],'frames':frames,'fps':30,'duration_seconds':seconds,
            'control_timestep':dt,'LQR_Q':Q.tolist(),'LQR_R':R.tolist(),'LQR_K':K.tolist(),
            'cart_reference':'x=0.55*sin(0.85*t); xdot=0.55*0.85*cos(0.85*t)',
            'camera_id':0,'mujoco_version':mujoco.__version__,'resolution':[1280,720],
            'mean_reward_diagnostic_only':float(np.mean(rewards)),'max_abs_pole_angle':float(np.max(np.abs(angles))),
            'source_url':'https://github.com/google-deepmind/dm_control'}
    (OUT/'dmc_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('WROTE DMC',record['max_abs_pole_angle'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('benchmark',choices=['libero','robomimic','dmc']);p.add_argument('--dataset',type=Path);p.add_argument('--demo',default='demo_0');p.add_argument('--limit',type=int);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.benchmark=='dmc':dmc()
    else:replay(args.benchmark,args.dataset or (LIBERO_DATA if args.benchmark=='libero' else SQUARE_DATA),args.demo,args.limit)
if __name__=='__main__':main()
