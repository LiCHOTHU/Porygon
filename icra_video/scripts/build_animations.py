#!/usr/bin/env python3
"""Original CAST teaching animations, rendered from a reproducible toy actor.

No robot measurements are generated here. The fixed synthetic critic isolates
actor optimization. Frames interpolate saved parameter updates for readability.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('CAST_VIDEO_ANIMATION_DIR', ROOT / 'animations'))
W, H = 1280, 720
BG = '#16273B'; PANEL = '#102135'; WHITE = '#F3F6FA'; MUTED = '#AFBDD0'
BLUE = '#7FB0ED'; TEAL = '#42D7B4'; ORANGE = '#F0B360'; GRAY = '#8D9BAC'; LINE = '#33455B'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
ACADEMIC = os.environ.get('CAST_VIDEO_THEME') == 'academic'
FILL_GREEN, FILL_ORANGE, REGION = '#25483F', '#514336', '#193C38'
if ACADEMIC:
    BG = '#FFFFFF'; PANEL = '#F3F4F5'; WHITE = '#202124'; MUTED = '#626B73'
    BLUE = '#356A96'; TEAL = '#287764'; ORANGE = '#B0602E'; GRAY = '#777F87'; LINE = '#CBD0D5'
    FILL_GREEN, FILL_ORANGE, REGION = '#E4EFE9', '#F4E8DC', '#F0F5F1'
PARAMS = dict(seed=285, dimensions=1, eta_Q=.18, delta_Q=2., eta_0=.12,
              eta_anc=.48, rho=.4, delta_total=.08, actor_lr=.35,
              actor_fit_steps=4, outer_updates=80, batch_size=512)
CLIPS = {
 'A01': ('A01_accumulated_drift',12,10),
 'A02': ('A02_paired_residuals',8,6),
 'A03': ('A03_cast_update',30,21),
 'A04': ('A04_regress_refresh',10,7.7),
 'A05': ('A05_measured_learning_progress',30,23),
}

@lru_cache(None)
def font(size=28, bold=False): return ImageFont.truetype(BOLD if bold else FONT, size)
def smooth(t): t=float(np.clip(t,0,1)); return t*t*(3-2*t)
def lerp(a,b,t): return np.asarray(a)*(1-t)+np.asarray(b)*t

def clip(v,cap):
    """Last axis is action dimension; independent clipping per candidate."""
    v=np.asarray(v); n=np.linalg.norm(v,axis=-1,keepdims=True)
    return v*np.minimum(1,cap/np.maximum(n,1e-12))

def simulate():
    torch.set_num_threads(1); torch.manual_seed(PARAMS['seed'])
    # Common random numbers make the two actor updates directly comparable.
    z=torch.randn(PARAMS['batch_size'],generator=torch.Generator().manual_seed(PARAMS['seed']),dtype=torch.float64)
    z=(z-z.mean())/z.std(unbiased=False)
    base=.27*z
    output={}
    for method in ['clipping_only','cast']:
        theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        opt=torch.optim.SGD([theta],lr=PARAMS['actor_lr'])
        hist=[theta.detach().numpy().copy()]; records=[]
        for k in range(PARAMS['outer_updates']):
            with torch.no_grad():
                r=theta[0]+theta[1]*z; a=base+r
                q=3-.22*(a-3.4)**2
                scale=q.abs().mean().clamp_min(1e-6)
                g=-.44*(a-3.4)/scale
                dq=PARAMS['eta_Q']*g.clamp(-PARAMS['delta_Q'],PARAMS['delta_Q'])
                dr=torch.zeros_like(r)
                if method=='cast':
                    dr=-PARAMS['eta_0']*r-PARAMS['eta_anc']*(1-PARAMS['rho']/r.abs().clamp_min(1e-6)).clamp_min(0)*r
                delta=(dq+dr).clamp(-PARAMS['delta_total'],PARAMS['delta_total'])
                target=(r+delta).detach()
            assert not target.requires_grad
            fitted=[theta.detach().numpy().copy()]; losses=[]
            for j in range(PARAMS['actor_fit_steps']):
                opt.zero_grad(); loss=((theta[0]+theta[1]*z-target)**2).mean()
                losses.append(float(loss.detach())); loss.backward(); opt.step()
                fitted.append(theta.detach().numpy().copy())
            losses.append(float(((theta[0]+theta[1]*z-target)**2).mean().detach()))
            assert max(abs(delta)).item()<=PARAMS['delta_total']+1e-12
            assert losses[-1]<=losses[0]+1e-12
            hist.append(theta.detach().numpy().copy())
            records.append(dict(theta=np.array(fitted),z=z.numpy(),base=base.numpy(),
                                target=(base+target).numpy(),losses=losses))
        output[method]=dict(history=np.array(hist),records=records)
    summary={
      'description':'Synthetic fixed-critic actor diagnostic; not benchmark training or a performance guarantee.',
      'base':'a0(z) = 0.27 z, z ~ Normal(0,1)',
      'actor':'a_theta(z) = a0(z) + b + w z; theta=(b,w)',
      'critic':'Q_hat(a) = 3 - 0.22 (a - 3.4)^2; held fixed in this diagnostic',
      'toy_useful_region':[-.25,1.15],
      'note':'Useful region is an illustrative assumption, not a reward learned or measured from a robot.',
      'parameters':PARAMS,
      'comparison':'Shared base, noise batch, critic, guidance scaling/cap, total cap and optimizer. Only the two restoring terms differ.',
      'histories':{m:v['history'].tolist() for m,v in output.items()},
      'checks':{'detached_targets':True,'target_step_cap':True,'fit_reduces_loss_each_round':True},
    }
    (OUT/'toy_trace.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert output['cast']['history'][-1,0]<1
    assert output['clipping_only']['history'][-1,0]>2.5
    return output

class Canvas:
    def __init__(self):
        self.im=Image.new('RGB',(W,H),BG); self.d=ImageDraw.Draw(self.im)
    def text(self,x,y,s,size=28,color=WHITE,bold=False,anchor=None):
        self.d.text((x,y),s,font=font(size,bold),fill=color,anchor=anchor)
    def line(self,points,color=LINE,width=2):
        self.d.line([tuple(map(float,p)) for p in points],fill=color,width=width,joint='curve')
    def dashed(self,a,b,color=BLUE,width=2,dash=9):
        a,b=np.array(a),np.array(b); length=np.linalg.norm(b-a)
        for n in np.arange(0,length,2*dash): self.line([lerp(a,b,n/length),lerp(a,b,min(n+dash,length)/length)],color,width)
    def arrow(self,a,b,color=TEAL,width=5,head=15):
        a,b=np.array(a,dtype=float),np.array(b,dtype=float)
        if np.linalg.norm(b-a)<1:return
        self.line([a,b],color,width); u=(b-a)/np.linalg.norm(b-a); v=np.array([-u[1],u[0]])
        self.d.polygon([tuple(b),tuple(b-head*u+.45*head*v),tuple(b-head*u-.45*head*v)],fill=color)
    def dot(self,p,color=TEAL,r=8,fill=True):
        x,y=p;self.d.ellipse((x-r,y-r,x+r,y+r),fill=color if fill else BG,outline=color,width=3)
    def square(self,p,color=TEAL,r=8):
        x,y=p; self.d.rectangle((x-r,y-r,x+r,y+r),fill=BG,outline=color,width=3)
    def pill(self,x,y,w,label,color=TEAL,size=24):
        if not ACADEMIC:
            self.d.rounded_rectangle((x,y,x+w,y+48),radius=12,fill=PANEL)
        self.text(x+w/2,y+24,label,size,color,True,'mm')
    def header(self,title,sub):
        self.text(35,25,title,34,WHITE,True);self.text(36,76,sub,23,MUTED)
    def footer(self,label='Illustrative action space · fixed observation · not a robot experiment'):
        self.text(W/2,692,label,20,MUTED,anchor='mm')
    def curve(self,x,y,color,width=4,fill=False):
        pts=list(zip(x,y))
        if fill:self.d.polygon(pts,fill=fill)
        self.line(pts,color,width)

def gaussian(c,mu,sigma,xx,yy,ww,hh,color,filled=False):
    vals=np.linspace(-1,4,400); sig=max(.001,abs(sigma))
    # Peak-normalized shapes preserve means and widths without clipping tall peaks.
    height=hh*np.exp(-.5*((vals-mu)/sig)**2)
    x=xx+(vals+1)/5*ww;y=yy-height
    if filled:
        poly=list(zip(x,y))+[(x[-1],yy),(x[0],yy)]
        c.d.polygon(poly,fill=FILL_GREEN if color==TEAL else FILL_ORANGE)
    c.line(zip(x,y),color,4)

def interpolate_history(hist,k):
    i=min(int(k),len(hist)-2); return lerp(hist[i],hist[i+1],min(k-i,1))

def drift(t,data):
    c=Canvas();c.header('Small steps can accumulate into large drift.','Same starting policy. Same critic. Same step cap.')
    # Shared action axis across the critic and both policy panels.
    xx,ww=225,995
    mapx=lambda a:xx+(a+1)/5*ww
    c.d.rounded_rectangle((mapx(-.25),145,mapx(1.15),608),radius=0 if ACADEMIC else 10,fill=REGION)
    c.text(mapx(.45),122,'Useful region in this toy',21,TEAL,anchor='mm')
    c.text(mapx(3.4),122,'Critic overestimates',21,ORANGE,anchor='mm')
    a=np.linspace(-1,4,300); q=3-.22*(a-3.4)**2
    c.line(zip(mapx(a),255-90*(q-q.min())/(q.max()-q.min())),ORANGE,4)
    c.text(35,188,'Predicted',24,ORANGE,True);c.text(35,219,'value',24,ORANGE,True)
    c.arrow((mapx(.8),234),(mapx(2.5),183),ORANGE,3,12)
    c.text(35,337,'Clipping',26,ORANGE,True);c.text(35,373,'only',26,ORANGE,True)
    c.text(35,514,'CAST',28,TEAL,True)
    k=80*smooth((t-.4)/7)
    for method,y,color in [('clipping_only',421,ORANGE),('cast',593,TEAL)]:
        c.line([(xx,y),(xx+ww,y)],LINE,2)
        p=interpolate_history(data[method]['history'],k)
        gaussian(c,p[0],.27+p[1],xx,y,ww,130,color,True)
        gaussian(c,0,.27,xx,y,ww,130,BLUE)
        c.dashed((mapx(0),y-119),(mapx(0),y+5),BLUE,1)
        if t>8:
            label='Keeps moving away' if method=='clipping_only' else 'Improves with a base reference'
            c.text(685,y-71,label,24,color,True)
    c.text(xx,625,'Frozen base',23,BLUE);c.text(745,625,'Action coordinate →',23,MUTED)
    c.footer('Toy actor · curve heights normalized for display · fixed critic')
    return c.im

def pairing(t,data):
    c=Canvas();c.header('Every action has its own frozen reference.','The same noise sample goes through both branches.')
    progress=smooth((t-.3)/3.2)
    c.pill(60,125,260,'Shared noise z',BLUE);c.pill(510,125,240,'Frozen base',BLUE);c.pill(940,125,280,'Current policy',TEAL)
    for i,z in enumerate([-1.2,0,1.2]):
        y=247+i*125;a0=.27*z
        learned=data['cast']['history'][20];r=(learned[0]+learned[1]*z)*progress
        c.pill(70,y-23,185,f'Noise sample {i+1}',MUTED,21)
        c.arrow((278,y),(423,y),BLUE,3,12)
        # Parallel action axes use identical scales; each row tracks the same z.
        left,right=444,1200;xm=lambda a:left+(a+.65)/1.8*(right-left)
        c.line([(left,y),(right,y)],LINE,2)
        bx,ax=xm(a0),xm(a0+r)
        c.dot((bx,y),BLUE,10)
        if r>.02:
            c.arrow((bx+13,y),(ax-13,y),TEAL,5,14);c.dot((ax,y),TEAL,10)
            c.text((bx+ax)/2,y-28,'learned residual',22,TEAL,anchor='mm')
    c.pill(173,601,934,'Current action = frozen base action + learned residual',WHITE,27)
    c.footer('One observation, three noise samples · references stay paired during training')
    return c.im

def geometry(t,data):
    c=Canvas();c.header('Build one target, then teach the actor to reach it.','Critic guidance + restoration + a cap on the combined step')
    # D=2 geometric illustration; separate from the D=1 trained actor.
    base=np.array([0.,0.]);r=np.array([.85,.28]);g=np.array([1.8,.95])
    rho=.38;radius=np.sqrt(2)*rho
    dq=.55*clip(g,1.2);weak=-.08*r;radial=-.28*max(1-radius/np.linalg.norm(r),0)*r
    combined=dq+weak+radial;total=clip(combined,.30)
    origin=np.array([180.,497.]);scale=425
    screen=lambda a:origin+scale*np.asarray(a)*[1,-1]
    b,cur=screen(base),screen(r);target=screen(r+total)
    # A partially shown soft radius avoids conflating it with an output bound.
    for ang in np.arange(0,2*np.pi,.13):
        p1=screen(base+radius*np.array([np.cos(ang),np.sin(ang)]))
        p2=screen(base+radius*np.array([np.cos(ang+.06),np.sin(ang+.06)]))
        if min(p1[0],p2[0])>35 and max(p1[1],p2[1])<605:c.line([p1,p2],BLUE,2)
    c.text(47,567,'Soft reference radius',23,BLUE)
    c.dashed(b,cur,BLUE,2);c.dot(b,BLUE,10);c.dot(cur,WHITE,10)
    c.text(b[0]-35,b[1]+23,'Base',25,BLUE,True)
    c.text(cur[0]-60,cur[1]+26,'Current',25,WHITE,True)
    if t<5:
        c.pill(464,139,748,'Start from the current action and its paired base.',WHITE,23)
    elif t<10:
        f=smooth((t-5)/2);end=screen(r+dq*f)
        c.arrow(cur,end,ORANGE,6,17)
        c.text(857,210,'Critic proposal',25,ORANGE,True)
        c.pill(400,139,812,'Normalize the critic gradient, cap it, then scale it.',ORANGE,23)
    elif t<17:
        c.arrow(cur,screen(r+dq),ORANGE,5,16)
        f=smooth((t-10)/2)
        c.arrow(cur,screen(r+weak*f),BLUE,6,12)
        if t>=13:
            # Head-to-tail construction; both fields were evaluated at current r.
            f2=smooth((t-13)/2)
            c.arrow(screen(r+weak),screen(r+weak+radial*f2),TEAL,6,14)
        c.text(845,210,'Critic proposal',24,ORANGE,True)
        c.text(43,171,'Weak pull',25,BLUE,True);c.text(43,208,'Always active',21,MUTED)
        c.text(43,266,'Radial pull',25,TEAL,True);c.text(43,303,'Stronger beyond radius',21,MUTED)
        c.pill(400,139,812,'Restoration responds to departure from the base.',TEAL,23)
    else:
        c.arrow(cur,screen(r+dq),ORANGE,4,14)
        c.arrow(screen(r+dq),screen(r+combined),TEAL,5,14)
        rr=.30*scale
        for ang in np.arange(0,2*np.pi,.14):
            c.line([cur+rr*np.array([np.cos(ang),np.sin(ang)]),cur+rr*np.array([np.cos(ang+.065),np.sin(ang+.065)])],GRAY,2)
        f=smooth((t-17)/2)
        c.arrow(cur,lerp(cur,target,f),TEAL,7,18);c.square(lerp(cur,target,f),TEAL,10)
        c.text(target[0]+23,target[1]+8,'Target',27,TEAL,True)
        c.text(831,474,'Step cap is centered',22,MUTED);c.text(831,506,'on the current action.',22,MUTED)
        c.pill(427,139,785,'Add the fields; cap the final displacement.',TEAL,24)
    c.pill(335,604,880,'A persistent pull toward the base, not a hard wall.',TEAL,25)
    c.footer('2D geometric example · restoration is evaluated at the current action')
    return c.im

def regression(t,data):
    c=Canvas();c.header('The target is fixed. The actor learns to match it.','Update the residual parameters, then construct fresh targets.')
    # Three consecutive recorded actor updates, around radial activation.
    elapsed=min(max(t-.3,0),9-1/30)
    round_index=min(int(elapsed//3),2);phase=(elapsed%3)*6.5/3
    record=data['cast']['records'][4+round_index]
    step=4*smooth((phase-1)/4)
    theta=interpolate_history(record['theta'],step)
    stages=['Construct targets','Fit the residual','Refresh targets']
    active=0 if phase<1 else 1 if phase<5.4 else 2
    for i,label in enumerate(stages):
        c.pill(44+i*409,121,372,label,TEAL if active==i else MUTED,24)
    plot=(165,218,1040,335);xx,yy,ww,hh=plot
    sx=lambda z:xx+(np.asarray(z)+2)/4*ww
    sy=lambda a:yy+hh-(np.asarray(a)+.6)/1.9*hh
    c.line([(xx,yy+hh),(xx+ww,yy+hh)],LINE,2)
    c.line([(xx,yy),(xx,yy+hh)],LINE,2)
    c.text(35,254,'Action',24,MUTED);c.text(1040,574,'Noise z →',23,MUTED)
    fraction=float(np.mean((record['base']+theta[0]+theta[1]*record['z']-record['target'])**2))/record['losses'][0]
    c.text(858,196,'Target-fitting error',20,MUTED)
    c.d.rounded_rectangle((859,229,1203,241),radius=0 if ACADEMIC else 4,fill=PANEL)
    if fraction>0.002:c.d.rounded_rectangle((859,229,859+344*fraction,241),radius=0 if ACADEMIC else 4,fill=TEAL)
    zz=np.linspace(-1.8,1.8,160)
    c.line(zip(sx(zz),sy(.27*zz)),BLUE,3)
    c.line(zip(sx(zz),sy(.27*zz+theta[0]+theta[1]*zz)),WHITE,5)
    inds=[int(np.abs(record['z']-z).argmin()) for z in np.linspace(-1.5,1.5,5)]
    for idx in inds:
        z=record['z'][idx];a=.27*z+theta[0]+theta[1]*z;tar=record['target'][idx]
        c.dashed((sx(z),sy(a)),(sx(z),sy(tar)),TEAL,2,5)
        c.square((sx(z),sy(tar)),TEAL,10);c.dot((sx(z),sy(a)),WHITE,6)
    c.line([(48,622),(91,622)],BLUE,4);c.text(102,607,'Frozen base',23,BLUE)
    c.line([(384,622),(427,622)],WHITE,4);c.text(438,607,'Actor output',23,WHITE)
    c.square((800,622),TEAL,9);c.text(824,607,'Fixed regression target',23,TEAL)
    c.footer('Actual gradient-descent updates of a small residual actor · targets are detached')
    return c.im

def evidence(t,data):
    c=Canvas();c.header('Restoration matters on the tested tasks.','LIBERO · mean success across five tasks · measured results')
    source=json.loads((ROOT/'assets/experiment_data.json').read_text())
    # Exact source keys are resolved by an explicit mapping; no learning curves inferred.
    comp=source['component_comparison']
    values=[np.mean([task[k] for task in comp['tasks']]) for k in ['base','none','clipping','radial','full']]
    names=['Base','No controls','Clipping only','Radial only','CAST']
    colors=[BLUE,GRAY,ORANGE,BLUE,TEAL]
    f=smooth((t-1)/4)
    for i,(name,val,color) in enumerate(zip(names,values,colors)):
        y=185+i*82
        c.text(44,y,name,26,color,name=='CAST')
        c.d.rounded_rectangle((305,y-1,1113,y+38),radius=0 if ACADEMIC else 4,fill=PANEL)
        if val>0:c.d.rounded_rectangle((305,y-1,305+808*val/100*f,y+38),radius=0 if ACADEMIC else 4,fill=color)
        else:
            c.line([(305,y),(305,y+38)],color,5)
        if t>5:c.text(1140,y+18,f'{val:.1f}%' if val else '0%',25,color,True,'mm')
    c.text(44,621,'Equal comparison: all five update configurations are shown.',24,WHITE)
    c.footer('Reported aggregate success · not a success-versus-training curve')
    return c.im

from figure2_animation import render as animated_figure2, write_trace as write_figure2_trace
DRAW={'A01':drift,'A02':pairing,'A03':animated_figure2,'A04':regression,'A05':evidence}

def render(ident,data,fps=30,posters_only=False):
    stem,duration,poster_time=CLIPS[ident]
    poster=DRAW[ident](poster_time,data)
    poster.save(OUT/'posters'/f'{stem}.png')
    if posters_only:return
    dest=OUT/f'{stem}.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(round(duration*fps)):
            proc.stdin.write(DRAW[ident](i/fps,data).tobytes())
            if i%(fps*5)==0:print(f'{ident}: {i/fps:g}/{duration}s',flush=True)
        proc.stdin.close();assert proc.wait()==0
    except BaseException:
        proc.kill();proc.wait();raise
    print(f'Wrote {dest.relative_to(ROOT)}',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clips',nargs='+',choices=CLIPS,default=list(CLIPS));parser.add_argument('--fps',type=int,default=30);parser.add_argument('--posters-only',action='store_true')
    args=parser.parse_args();(OUT/'posters').mkdir(parents=True,exist_ok=True)
    data=simulate();write_figure2_trace(OUT/'figure2_trace.json')
    for ident in args.clips:render(ident,data,args.fps,args.posters_only)
    # Consistent review sheet, including geometry's intermediate restoring phase.
    keys=[('A01',2),('A01',10),('A02',6),('A03',9),('A03',15.5),('A03',21),('A03',26),('A04',7.7),('A05',23)]
    sheet=Image.new('RGB',(1280,1080),'#0D1827')
    for i,(key,t) in enumerate(keys):
        frame=DRAW[key](t,data);frame.thumbnail((426,240));sheet.paste(frame,((i%3)*426,(i//3)*360+40))
        ImageDraw.Draw(sheet).text(((i%3)*426+12,(i//3)*360+8),f'{key} / {t}s',font=font(22),fill=WHITE)
    sheet.save(OUT/'contact_sheet.jpg',quality=94)

if __name__=='__main__':main()
