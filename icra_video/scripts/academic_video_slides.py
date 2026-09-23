#!/usr/bin/env python3
"""Plain academic slide backgrounds for the CAST explanation cut."""
from pathlib import Path
import io
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'explanation'
BG='#FFFFFF';INK='#202124';MUTED='#626B73';LINE='#D5D9DD'
BLUE='#356A96';GREEN='#287764';ORANGE='#B0602E'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)


class Slide:
    def __init__(self,title=None):
        self.im=Image.new('RGB',(1920,1080),BG)
        self.d=ImageDraw.Draw(self.im)
        if title:
            self.text(80,55,title,52,bold=True,width=1760)
            self.d.line((80,145,1840,145),fill=LINE,width=2)
    def text(self,x,y,text,size=30,color=INK,bold=False,width=None):
        f=font(size,bold)
        lines=[]
        for paragraph in text.split('\n'):
            current=''
            for word in paragraph.split():
                candidate=(current+' '+word).strip()
                if width and self.d.textlength(candidate,font=f)>width and current:
                    lines.append(current);current=word
                else:current=candidate
            lines.append(current)
        for i,line in enumerate(lines):
            if width:assert self.d.textlength(line,font=f)<=width,(line,width)
            self.d.text((x,y+i*size*1.35),line,font=f,fill=color)
        return y+len(lines)*size*1.35
    def equation(self,expr,x,y,width,height):
        stream=io.BytesIO()
        with matplotlib.rc_context({'mathtext.fontset':'stix','savefig.transparent':True}):
            math_to_image('$'+expr+'$',stream,dpi=220,color=INK,format='png')
        im=Image.open(stream).convert('RGBA');im=im.crop(im.getchannel('A').getbbox())
        scale=min(width/im.width,height/im.height)
        im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
        self.im.paste(im,(int(x),int(y)),im)
    def image(self,path,box):
        x,y,w,h=box
        im=Image.open(path).convert('RGB')
        factor=min(w/im.width,h/im.height)
        im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.LANCZOS)
        self.im.paste(im,(round(x+(w-im.width)/2),round(y+(h-im.height)/2)))
    def arrow(self,a,b,color=INK):
        self.d.line((a,b),fill=color,width=3)
        ang=math.atan2(b[1]-a[1],b[0]-a[0]);r=14
        self.d.polygon([b,(b[0]-r*math.cos(ang-.4),b[1]-r*math.sin(ang-.4)),(b[0]-r*math.cos(ang+.4),b[1]-r*math.sin(ang+.4))],fill=color)
    def box(self,box,title,sub,color=INK):
        x,y,w,h=box
        self.d.rectangle((x,y,x+w,y+h),outline=LINE,width=2)
        self.text(x+22,y+17,title,31,color,bold=True)
        self.text(x+22,y+62,sub,24,MUTED)


POSITIONS={
 'A01':[60,165,1440,810],
 'A02':[850,300,1000,562],
 'A03':[0,0,1920,1080],
 'A04':[65,195,1230,692],
 'A05':[60,245,1080,608],
}


def build():
    posters=OUT/'posters';posters.mkdir(parents=True,exist_ok=True)
    scenes={}
    s=Slide();s.text(80,100,'CAST',76,bold=True)
    s.text(80,235,'Anchoring critic-guided refinement\nof generative robot policies',43,width=1000)
    s.text(80,440,'Improve a pretrained skill while keeping\nthe original policy as a reference.',32,MUTED,width=1000)
    s.equation(r'a_\theta(s,z)=\pi_0(s,z)+r_\theta(s,z)',80,650,930,86)
    s.text(84,772,'Frozen base + learned correction',30,MUTED)
    s.image(ROOT/'assets/robot/jigglypuff_initial_chest.png',(1130,205,690,475))
    s.text(1130,704,'OpenArm manipulation setup',27,bold=True)
    s.text(1130,750,'Demonstration image',23,MUTED)
    scenes['01']=s

    s=Slide('Small updates can still accumulate drift.')
    s.text(1540,365,'Clipping',31,ORANGE,bold=True,width=300)
    s.text(1540,419,'Limits each\nindividual step.',28,MUTED,width=300)
    s.text(1540,615,'Restoration',31,GREEN,bold=True,width=300)
    s.text(1540,669,'Responds to\naccumulated\ndeparture.',28,MUTED,width=300)
    scenes['02']=s

    s=Slide('Keep the base fixed. Learn a correction.')
    s.equation(r'a_\theta(s,z)=\pi_0(s,z)+r_\theta(s,z)',90,192,1350,95)
    s.text(110,400,'Observation s + shared noise z',28,bold=True,width=680)
    s.box((105,487,535,114),'Pretrained base policy','Fixed during refinement',BLUE)
    s.box((105,691,535,114),'Residual policy','Learned during refinement',GREEN)
    s.arrow((375,455),(375,480),MUTED)
    s.d.line((75,453,75,750),fill=LINE,width=2);s.arrow((75,750),(100,750),MUTED)
    s.text(102,878,'Each action keeps its paired reference.',27,MUTED,width=670)
    scenes['03']=s

    # The method animation fills this slide with its own equations and four-panel layout.
    scenes['04']=Slide()

    s=Slide('Learn from fixed targets, then refresh.')
    s.text(1375,242,'Soft restoration',33,bold=True,width=455)
    s.text(1375,306,'The pull increases\nwith departure.',27,MUTED,width=440)
    x,y,w,h=1410,710,345,260
    s.d.line((x,y-h,x,y,x+w,y),fill=MUTED,width=2)
    values=np.linspace(0,1,150);rho=.35
    potential=.15*values**2+1.3*np.maximum(values-rho,0)**2
    points=[(x+w*v,y-h*p/potential[-1]) for v,p in zip(values,potential)]
    s.d.line(points,fill=GREEN,width=4)
    rx=x+w*rho
    for yy in range(y-h,y,15):s.d.line((rx,yy,rx,min(yy+7,y)),fill=LINE,width=2)
    s.equation(r'U(r)',1362,402,90,45)
    s.text(x+95,y+17,'radius',22,MUTED)
    s.text(x+40,y+69,'Residual magnitude',23,MUTED)
    s.text(80,933,'Fixed targets simplify fitting; soft restoration leaves room to adapt.',28,MUTED,width=1750)
    scenes['05']=s

    data=json.loads((ROOT/'assets/experiment_data.json').read_text())
    means={k:float(np.mean(v))/1000 for k,v in data['diffusion_efficiency']['environment_steps'].items()}
    s=Slide('Simulation evidence')
    s.text(1215,209,'robomimic square',34,bold=True,width=625)
    s.text(1215,263,'Steps to 90% success',28,MUTED,width=625)
    x,y,w=1380,401,350
    for i,(name,color) in enumerate([('CAST',GREEN),('DICE-RL',BLUE)]):
        yy=y+i*110
        s.text(1215,yy+8,name,28,color,bold=True)
        length=w*means[name]/300
        s.d.rectangle((x,yy,x+length,yy+47),fill=color)
        s.text(x+length+12,yy+8,f'{means[name]:.1f}K',26)
    ay=y+175;s.d.line((x,ay,x+w,ay),fill=LINE,width=2)
    for val in [0,100,200,300]:
        xx=x+w*val/300;s.d.line((xx,ay,xx,ay+7),fill=MUTED,width=2)
        s.text(xx-12,ay+15,str(val),21,MUTED)
    s.text(1375,ay+58,'Environment steps (thousands)',23,MUTED,width=435)
    gain=100*(1-means['CAST']/means['DICE-RL'])
    s.text(1215,734,f'{gain:.0f}% fewer steps with CAST',30,GREEN,bold=True,width=625)
    s.text(1215,792,'Means across three runs;\nsame diffusion base.',24,MUTED,width=625)
    s.text(80,935,'Restoration prevents the collapse seen with clipping alone on these five LIBERO tasks.',27,MUTED,width=1750)
    scenes['06']=s

    s=Slide();s.text(80,150,'Improve the skill.\nKeep a reference.',66,bold=True,width=1740)
    for x,title,desc,color in [(80,'Critic guidance','Propose an improvement.',ORANGE),(680,'Restoration','Pull toward the frozen base.',GREEN),(1280,'Step control','Limit each target step.',BLUE)]:
        s.text(x,527,title,34,color,bold=True,width=535)
        s.d.line((x,590,x+505,590),fill=LINE,width=2)
        s.text(x,625,desc,28,MUTED,width=535)
    s.text(80,902,'cast2027.github.io',27,MUTED)
    scenes['10']=s
    for ident,s in scenes.items():s.im.save(posters/f'slide-{ident}.png')
    (OUT/'assets/academic_layout.json').write_text(json.dumps(POSITIONS,indent=2)+'\n')
    return {key:posters/f'slide-{key}.png' for key in scenes}


if __name__=='__main__':
    build()
    print('Built seven academic slide backgrounds.')
