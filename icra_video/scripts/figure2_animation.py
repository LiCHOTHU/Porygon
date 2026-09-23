#!/usr/bin/env python3
"""Animate the submitted Figure 2 from its original numerical arrays.

Overview preserves the clockwise layout. Focus shots keep the same action-space
scale, show one equation at a time, and compute every clipping endpoint exactly.
"""
from pathlib import Path
from functools import lru_cache
import io,json,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
matplotlib.rcParams['mathtext.fontset']='stix'

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'_ICRA_2027__CAST/figures'
CFG=json.loads((SOURCE/'method_update.json').read_text())['configuration']
with np.load(SOURCE/'method_update.npz') as f:DATA={k:f[k].copy() for k in f.files}
W,H=1280,720
BG='#0D1827';CARD='#16273B';MUTED='#AFBDD0';WHITE='#F3F6FA'
BLUE='#3667A6';GRAY='#82909D';ORANGE='#D55E00';TEAL='#008577';INK='#283746'
LIGHT='#FBFCFD';BORDER='#CED6DE';ACCENT='#42D7B4'
ACADEMIC=os.environ.get('CAST_VIDEO_THEME')=='academic'
if ACADEMIC:
    BG=CARD=LIGHT='#FFFFFF';WHITE=INK='#202124';MUTED='#626B73'
    BLUE='#356A96';GRAY='#777F87';ORANGE='#B0602E';TEAL=ACCENT='#287764';BORDER='#D5D9DD'
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';FB='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
STAGES=[('1  Sample actions',0,5),('2  Critic correction',5,11),('3  Restore + clip',11,24),('4  Fit the residual',24,30)]

@lru_cache(None)
def font(size,bold=False):return ImageFont.truetype(FB if bold else F,size)
def ease(t):t=float(np.clip(t,0,1));return t*t*(3-2*t)
def blend(a,b,t):return np.asarray(a)*(1-t)+np.asarray(b)*t

def txt(im,x,y,text,size=24,color=WHITE,bold=False,anchor=None):
    ImageDraw.Draw(im).text((x,y),text,font=font(size,bold),fill=color,anchor=anchor)

@lru_cache(256)
def equation(expr,size=32,color=WHITE):
    stream=io.BytesIO()
    with matplotlib.rc_context({'savefig.transparent':True}):
        math_to_image('$'+expr+'$',stream,dpi=150,color=color,format='png')
    img=Image.open(stream).convert('RGBA');alpha=img.getchannel('A');box=alpha.getbbox()
    if box:img=img.crop(box)
    # Set the rendered cap-height consistently while retaining fraction height.
    # math_to_image uses a 10-point font; size is a target relative font scale.
    scale=size/(10*150/72)
    return img.resize((max(1,round(img.width*scale)),max(1,round(img.height*scale))),Image.Resampling.LANCZOS)

def eq(im,x,y,expr,size=32,color=WHITE,max_width=None):
    img=equation(expr,size,color)
    if max_width and img.width>max_width:
        f=max_width/img.width;img=img.resize((round(img.width*f),round(img.height*f)),Image.Resampling.LANCZOS)
    im.paste(img,(round(x),round(y)),img)
    return img.height

def arrow(draw,a,b,color,width=4,dashed=False,head=12):
    a,b=np.array(a),np.array(b);length=np.linalg.norm(b-a)
    if length<1:return
    if dashed:
        for i in np.arange(0,length,14):draw.line([tuple(blend(a,b,i/length)),tuple(blend(a,b,min(i+7,length)/length))],fill=color,width=width)
    else:draw.line([tuple(a),tuple(b)],fill=color,width=width)
    u=(b-a)/length;v=np.array([-u[1],u[0]]);head=min(head,max(4,.4*length))
    draw.polygon([tuple(b),tuple(b-head*u+.43*head*v),tuple(b-head*u-.43*head*v)],fill=color)

def disk(draw,p,color,r=6,hollow=False,diamond=False):
    x,y=p
    if diamond:draw.polygon([(x,y-r),(x+r,y),(x,y+r),(x-r,y)],fill=color)
    else:draw.ellipse((x-r,y-r,x+r,y+r),fill=LIGHT if hollow else color,outline=color if hollow else LIGHT,width=2)

def cloud(im,xy,std,color,transform,alpha=1,dashed=False,particles=None):
    mean=transform(xy);s=transform.scale;d=ImageDraw.Draw(im)
    layer=Image.new('RGBA',im.size);ld=ImageDraw.Draw(layer)
    rgb=tuple(int(color[i:i+2],16) for i in (1,3,5))
    # Nested transparent ellipses approximate the Gaussian shading of Figure 2.
    for rad in np.linspace(2.45,.1,24):
        rx,ry=std*rad*s
        ld.ellipse((mean[0]-rx,mean[1]-ry,mean[0]+rx,mean[1]+ry),fill=rgb+(round(3*alpha),))
    im.paste(Image.alpha_composite(Image.new('RGBA',im.size),layer),(0,0),layer)
    for rad,width in [(2.45,2),(1.177,3)]:
        rx,ry=std*rad*s;box=(mean[0]-rx,mean[1]-ry,mean[0]+rx,mean[1]+ry)
        if dashed:
            for a in range(0,360,22):d.arc(box,a,a+13,fill=color,width=width)
        else:d.ellipse(box,outline=color,width=width)
    if particles is not None:
        for p in particles[1:8]:disk(d,transform(p),color,3)

class Transform:
    def __init__(self,box):
        x,y,w,h=box;self.scale=min(w/1.86,h/1.24)
        self.origin=np.array([x+(w-1.86*self.scale)/2,y+(h-1.24*self.scale)/2])
    def __call__(self,p):return self.origin+self.scale*(np.asarray(p)-[-1.02,1.0])*[1,-1]

def geometric_panel(im,box,stage,progress=1,detail=True,subphase=None):
    """Render one panel; all panels use the same coordinate bounds and scale."""
    d=ImageDraw.Draw(im);T=Transform(box);std=np.asarray(CFG['base_std'])
    b,c,q,raw,comb,target=[DATA[k][0] for k in ['base','current','q_proposal','raw_q','combined','target']]
    factor=ease(progress)
    actor=blend(b,c,factor) if stage==0 else c
    actor_points=DATA['current']+(actor-c)
    cloud(im,b,std,BLUE,T,dashed=True,particles=DATA['base'])
    cloud(im,actor,std,GRAY,T,particles=actor_points)
    disk(d,T(b),BLUE,7 if detail else 4);disk(d,T(actor),INK,7 if detail else 4)
    if detail:
        eq(im,T(b)[0]-30,T(b)[1]+20,r'a_{0,i}',29,BLUE)
        eq(im,T(actor)[0]+20,T(actor)[1]+22,r'a_i',29,GRAY)
    if stage==0:
        arrow(d,T(b),T(actor),GRAY,4 if detail else 2)
        if detail:
            eq(im,(T(b)[0]+T(actor)[0])/2-14,T(actor)[1]-36,r'\bar r_i',31,GRAY)
            txt(im,T(b)[0],T(b)[1]-72,'Frozen base',23,BLUE,anchor='mm')
            txt(im,T(c)[0]+24,T(c)[1]-93,'Current policy',23,GRAY,anchor='mm')
    elif stage==1:
        # First reveal uncapped guidance, then visibly contract it onto d^Q.
        u=float(progress)
        if u<.4:
            endpoint=blend(c,raw,ease(u/.4));arrow(d,T(c),T(endpoint),ORANGE,4,dashed=True)
        else:
            f=ease((u-.4)/.35);endpoint=blend(raw,q,f)
            arrow(d,T(c),T(raw),ORANGE,2,dashed=True)
            cloud(im,endpoint,std,ORANGE,T);arrow(d,T(c),T(endpoint),ORANGE,5 if detail else 2)
            disk(d,T(endpoint),ORANGE,6)
        if detail:
            eq(im,T(raw)[0]-190,T(raw)[1]-35,r'\eta_Q g_i\ \mathrm{(raw)}',27,ORANGE)
            eq(im,T(q)[0]+18,T(q)[1]+17,r'd_i^Q',31,ORANGE)
    elif stage==2:
        cloud(im,q,std,ORANGE,T);disk(d,T(q),ORANGE,5);arrow(d,T(c),T(q),ORANGE,4 if detail else 2)
        # progress 0..0.5 restores; progress 0.5..1 clips the combined vector.
        restore=ease(min(float(progress)*2,1));end=blend(q,comb,restore)
        if subphase is not None:
            weak=-CFG['eta_0']*(c-b)
            end=q+weak*subphase[0]+(DATA['dr'][0]-weak)*subphase[1]
            middle=q+weak*subphase[0]
            arrow(d,T(q),T(middle),BLUE,5,head=6)
            arrow(d,T(middle),T(end),TEAL,5)
        else:arrow(d,T(q),T(end),TEAL,5 if detail else 2)
        disk(d,T(end),GRAY,5,hollow=True)
        if detail:
            name=r'd_i^{\mathrm{weak}}' if subphase is not None and subphase[1]==0 else r'd_i^R'
            eq(im,(T(q)[0]+T(comb)[0])/2-22,T(q)[1]-45,name,31,TEAL)
        if progress>=.5:
            center=T(c);radius=CFG['total_cap']*T.scale
            for angle in range(0,360,15):d.arc((center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius),angle,angle+5,fill=INK,width=3 if detail else 1)
            shrink=ease((float(progress)-.5)/.35);clipped=blend(comb,target,shrink)
            arrow(d,T(comb),T(target),GRAY,2,dashed=True)
            arrow(d,T(c),T(clipped),TEAL,5 if detail else 2)
            disk(d,T(clipped),TEAL,9 if detail else 5,diamond=True)
            if shrink>.97:cloud(im,target,std,TEAL,T)
            if detail:
                arrow(d,(T(c)[0]+radius+87,T(c)[1]+59),(T(c)[0]+radius,T(c)[1]),GRAY,2,head=7)
                eq(im,T(c)[0]+radius+35,T(c)[1]+65,r'\delta_{\mathrm{tot}}',29,GRAY)
                eq(im,T(target)[0]-25,T(target)[1]+105,r'\tilde a_i',31,TEAL)
                d.line([tuple(T(target)+[0,99]),tuple(T(target)+[0,14])],fill=TEAL,width=2)
                txt(im,box[0]+22,box[1]+25,'Cap around the current action',22,GRAY)
    elif stage==3:
        cloud(im,target,std,TEAL,T)
        # Actual gradient descent for a 2D translation residual, starting at the
        # paper's current residual and fitting exactly the paper's fixed targets.
        p=np.clip(progress,0,1)*8;k=min(int(p),7);theta=blend(FIT[k],FIT[k+1],min(p-k,1))
        current=DATA['base']+theta
        # Clear the old current cloud visually by marking moving outputs in ink.
        for idx in [0,2,6,8]:
            arrow(d,T(current[idx]),T(DATA['target'][idx]),TEAL,3 if detail else 1,head=9)
            disk(d,T(DATA['target'][idx]),TEAL,6 if detail else 3,diamond=True)
            disk(d,T(current[idx]),INK,5 if detail else 3)
        if detail:
            txt(im,box[0]+24,box[1]+25,'Actor outputs move; targets stay fixed.',23,TEAL)
            eq(im,T(target)[0]-73,T(target)[1]-54,r'\tilde a_i',31,TEAL)
    return im

# Derive a reproducible fitting trace for the last stage, using the exact arrays.
target_res=DATA['target']-DATA['base'];theta=(DATA['current']-DATA['base']).mean(0)
FIT=[theta.copy()];LOSSES=[]
for _ in range(8):
    error=theta[None,:]-target_res;LOSSES.append(float(np.mean(error**2)))
    grad=2*error.mean(0)/CFG['dimension'];theta=theta-.35*grad;FIT.append(theta.copy())
LOSSES.append(float(np.mean((theta[None,:]-target_res)**2)))
FIT=np.array(FIT)
assert all(b<a for a,b in zip(LOSSES,LOSSES[1:]))

# Visible step and equation cues; animation completion is never a long empty hold.
CUES=[
 (0,2,'overview','Clockwise roadmap from the submitted Figure 2'),
 (2,5,'sample','Pair base and current samples with the same noise'),
 (5,8,'raw','Show the normalized critic request'),
 (8,11,'critic_cap','Contract the raw request to the critic cap'),
 (11,14,'weak','Weak restoration is active at every nonzero residual'),
 (14,18,'radial','Additional restoration activates outside the RMS radius'),
 (18,24,'total_cap','Clip the combined update around the current action'),
 (24,28,'fit','Fit detached targets using the actor MSE'),
 (28,30,'recap','Return to the complete clockwise figure'),
]

def overview(t,complete=False):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    txt(im,32,18,'CAST · sample, propose, constrain, fit',32,WHITE,True)
    panels=[(28,75),(665,75),(665,378),(28,378)]
    math=[r'a_{0,i}=\pi_0(s,z_i),\quad a_i=a_{0,i}+\bar r_i',r'd_i^Q=\eta_Q C_{\delta_Q}(g_i)',r'\tilde r_i=\operatorname{sg}[\bar r_i+C_{\delta_{\mathrm{tot}}}(d_i^Q+d_i^R)]',r'\min_\theta\;\frac{1}{KD}\sum_i\|r_\theta(s,z_i)-\tilde r_i\|_2^2']
    for j,(x,y) in enumerate(panels):
        d.rounded_rectangle((x,y,x+588,y+283),radius=0 if ACADEMIC else 12,fill=LIGHT,outline=ACCENT if (j==0 and not complete) else BORDER,width=3 if (j==0 and not complete) else 1)
        txt(im,x+16,y+10,STAGES[j][0],24,INK,True)
        geometric_panel(im,(x+18,y+36,550,180),j,1,False)
        d.rounded_rectangle((x+10,y+221,x+578,y+273),radius=0 if ACADEMIC else 8,fill='#F0F3F6')
        eq(im,x+21,y+232,math[j],25,INK,max_width=548)
    for a,b in [((622,208),(657,208)),((960,359),(960,376)),((657,518),(622,518))]:arrow(d,a,b,MUTED,3,head=10)
    return im

def render(t,_data=None):
    if t<2:return overview(t)
    if t>=28:return overview(t,True)
    stage=0 if t<5 else 1 if t<11 else 2 if t<24 else 3
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    txt(im,32,22,'CAST · animated Figure 2',30,WHITE,True)
    # A compact clockwise map keeps the paper's panel ordering visible in focus.
    for j,(x,y) in enumerate([(1125,15),(1192,15),(1192,54),(1125,54)]):
        d.rounded_rectangle((x,y,x+54,y+29),radius=0 if ACADEMIC else 5,fill=ACCENT if j==stage else CARD)
        txt(im,x+27,y+14,str(j+1),18,BG if j==stage else MUTED,True,'mm')
    txt(im,32,76,STAGES[stage][0],34,ACCENT,True)
    d.rounded_rectangle((28,137,713,638),radius=0 if ACADEMIC else 14,fill=LIGHT,outline=BORDER,width=1)
    box=(42,164,660,447)
    if stage==0:progress=(t-2)/1.2
    elif stage==1:progress=(t-5)/6
    elif stage==2:progress=(t-11)/14 if t<18 else .5+.5*(t-18)/4
    else:progress=ease((t-24)/2.5)
    subphase=(ease((t-11)/1.1),ease((t-14)/1.7)) if stage==2 else None
    geometric_panel(im,box,stage,float(np.clip(progress,0,1)),subphase=subphase)
    x=749;mw=495
    if stage==0:
        txt(im,x,153,'One reference for each sample',24,WHITE,True)
        eq(im,x,221,r'a_{0,i}=\pi_0(s,z_i)',38,max_width=mw)
        eq(im,x,294,r'a_i=a_{0,i}+\bar r_i',38,max_width=mw)
        eq(im,x,376,r'\bar r_i=\operatorname{sg}[r_\theta(s,z_i)]',32,max_width=mw)
        txt(im,x,477,'Same observation s and noise zᵢ.',23,MUTED)
        txt(im,x,527,'The base stays frozen.',26,ACCENT,True)
    elif stage==1:
        txt(im,x,153,'Normalize → cap → scale',26,WHITE,True)
        eq(im,x,215,r'g_i=\frac{\nabla_a\bar Q_\phi(s,z_i,a_i)}{q_{\mathrm{scale}}}',36,max_width=mw)
        eq(im,x,319,r'd_i^Q=\eta_Q C_{\delta_Q}(g_i)',39,max_width=mw)
        eq(im,x,423,r'C_\delta(v)=v\min\!\left(1,\frac{\delta}{\|v\|_2}\right)',31,max_width=mw)
        eq(im,x,531,r'\|d_i^Q\|_2\leq\eta_Q\delta_Q',34,ACCENT,max_width=mw)
        txt(im,x,594,'Shrink the length; keep the direction.',21,MUTED)
    elif stage==2 and t<18:
        txt(im,x,153,'Restore toward the paired base',24,WHITE,True)
        eq(im,x,218,r'd_i^R=-\eta_0\bar r_i' if t>=14 else r'd_i^{\mathrm{weak}}=-\eta_0\bar r_i',39,ACCENT,max_width=mw)
        txt(im,x,282,'Weak pull at every nonzero residual',22,MUTED)
        if t>=14:
            eq(im,x+8,339,r'-\eta_{\mathrm{anc}}\left(1-\frac{\rho}{\max(m_i,\epsilon)}\right)_+\bar r_i',34,ACCENT,max_width=mw-8)
            txt(im,x,424,'Additional pull when mᵢ > ρ',24,MUTED)
            eq(im,x,478,r'm_i=\frac{\|\bar r_i\|_2}{\sqrt{D}}',34,max_width=mw)
        txt(im,x,581,'Soft restoration leaves room to adapt.',22,WHITE)
    elif stage==2:
        txt(im,x,153,'Limit the combined target step',24,WHITE,True)
        eq(im,x,223,r'\Delta_i=C_{\delta_{\mathrm{tot}}}(d_i^Q+d_i^R)',36,max_width=mw)
        eq(im,x,320,r'\tilde r_i=\operatorname{sg}[\bar r_i+\Delta_i]',36,max_width=mw)
        eq(im,x,415,r'\|\tilde r_i-\bar r_i\|_2\leq\delta_{\mathrm{tot}}',37,ACCENT,max_width=mw)
        txt(im,x,523,'The circle limits this target step.',23,MUTED)
        txt(im,x,565,'Its center is the current action.',23,WHITE,True)
    else:
        txt(im,x,153,'Learn from fixed targets',26,WHITE,True)
        eq(im,x,221,r'\mathcal{L}_{\mathrm{actor}}=',38,max_width=mw)
        eq(im,x,287,r'\mathbb{E}_s\!\left[\frac{1}{KD}\sum_{i=1}^{K}\|r_\theta(s,z_i)-\tilde r_i\|_2^2\right]',33,max_width=mw)
        eq(im,x,409,r'\tilde a_i=a_{0,i}+\tilde r_i',35,ACCENT,max_width=mw)
        txt(im,x,492,'sg: hold the target fixed.',24,MUTED)
        txt(im,x,540,'Fit the residual, then refresh.',24,WHITE,True)
    txt(im,32,657,'Same numerical construction as paper Figure 2 · illustrative action space',19,MUTED)
    return im

def write_trace(destination=None):
    p=Path(destination) if destination else ROOT/'animations/figure2_trace.json'
    trace={'source_npz':'../_ICRA_2027__CAST/figures/method_update.npz','source_json':'../_ICRA_2027__CAST/figures/method_update.json','duration_seconds':30,'cues':[dict(start=a,end=b,stage=c,explanation=d) for a,b,c,d in CUES], 'fitting':{'actor':'r_theta(s,z)=theta in this 2D schematic','learning_rate':.35,'theta':FIT.tolist(),'loss':LOSSES},'cap_center':'current action','note':'Original figure vectors are reused. Fitting is an illustrative translation actor trained to the figure targets, not benchmark evidence.'}
    p.write_text(json.dumps(trace,indent=2)+'\n')
