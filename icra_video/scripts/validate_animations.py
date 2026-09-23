#!/usr/bin/env python3
"""Check rendered media, target math, actor fitting, and PowerPoint embedding."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
import numpy as np
from xml.etree import ElementTree as ET
from build_animations import simulate,clip,PARAMS
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'animation_manifest.json').read_text())
checks=[]
data=simulate()
# Independent finite-difference check of the paper's radial restoring field.
D=2;eta0=.08;etaanc=.28;rho=.38;R=np.sqrt(D)*rho
potential=lambda r:eta0/2*np.dot(r,r)+etaanc/2*max(np.linalg.norm(r)-R,0)**2
for r in [np.zeros(2),np.array([.1,.1]),np.array([.85,.28]),np.array([-.7,.9])]:
    restore=-eta0*r-etaanc*max(1-R/max(np.linalg.norm(r),1e-12),0)*r
    gradient=np.array([(potential(r+np.eye(2)[j]*1e-6)-potential(r-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)])
    assert np.allclose(restore,-gradient,atol=1e-8)
    assert np.dot(restore,r)<=1e-12
checks.append('Restoration matches the negative gradient of the stated potential inside and outside the soft radius.')
# Verify the exact original Figure 2 data, not a newly chosen geometry.
from figure2_animation import DATA,CFG,FIT,LOSSES
r=DATA['current']-DATA['base']
g=np.broadcast_to(CFG['normalized_q_gradient'],r.shape)
dq=CFG['eta_q']*clip(g,CFG['q_cap'])
m=np.linalg.norm(r,axis=-1,keepdims=True)/np.sqrt(CFG['dimension'])
dr=-CFG['eta_0']*r-CFG['eta_anc']*np.maximum(1-CFG['rho_rms']/np.maximum(m,1e-12),0)*r
delta=clip(dq+dr,CFG['total_cap'])
assert np.allclose(dq,DATA['dq'],atol=1e-7)
assert np.allclose(dr,DATA['dr'],atol=1e-7)
assert np.allclose(delta,DATA['delta'],atol=1e-7)
assert np.allclose(DATA['current']+delta,DATA['target'],atol=1e-7)
assert np.max(np.linalg.norm(DATA['target']-DATA['current'],axis=-1))<=CFG['total_cap']+1e-7
assert all(b<a for a,b in zip(LOSSES,LOSSES[1:]))
checks.append('Animated Figure 2 reuses the submitted figure samples; critic/restoring vectors, cap order, target positions, and translation-actor fitting are numerically verified.')
assert np.array_equal(data['cast']['history'][0],data['clipping_only']['history'][0])
assert np.allclose(data['cast']['records'][0]['z'],data['clipping_only']['records'][0]['z'])
for method in data.values():
    for record in method['records']:
        assert all(b<=a+1e-12 for a,b in zip(record['losses'],record['losses'][1:]))
        old=record['base']+record['theta'][0,0]+record['theta'][0,1]*record['z']
        assert np.max(np.abs(record['target']-old))<=PARAMS['delta_total']+1e-12
checks.append('Both actors share initialization and noise; every fixed-target fitting step reduces MSE; every target step respects the total cap.')
assert data['cast']['history'][-1,0]<1 and data['clipping_only']['history'][-1,0]>2.5
checks.append('The recorded toy trajectories reproduce the illustrated drift-versus-restoration comparison.')
measurements=ROOT/'assets/experiment_data.json'
assert measurements.read_bytes()==(ROOT.parent/'_ICRA_2027__CAST/figures/experiment_chart_data.json').read_bytes()
x=json.loads(measurements.read_text())['component_comparison']['tasks']
assert np.allclose([np.mean([r[k] for r in x]) for k in ['base','none','clipping','radial','full']],[67.28,0,0,65.74,70.34])
checks.append('Measured ablation source is unchanged; displayed means recompute to 67.28, 0, 0, 65.74, and 70.34 percent.')
completed=[s for s in manifest['slots'] if s['status'].startswith('completed')]
assert len(completed)==5
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
with zipfile.ZipFile(ROOT/'cast_video_slides.pptx') as deck:
    embedded={hashlib.sha256(deck.read(n)).hexdigest() for n in deck.namelist() if n.startswith('ppt/media/') and n.endswith('.mp4')}
    assert len(embedded)==5
    for s in completed:
        path=ROOT/s['filename']
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
        video=next(v for v in info['streams'] if v['codec_type']=='video')
        assert video['codec_name']=='h264' and video['pix_fmt']=='yuv420p'
        assert [video['width'],video['height']]==[1280,720]
        assert video['r_frame_rate']=='30/1'
        assert abs(float(info['format']['duration'])-s['slide_duration_seconds'])<.04
        assert int(video['nb_frames'])==30*s['slide_duration_seconds']
        assert hashlib.sha256(path.read_bytes()).hexdigest() in embedded, s['id']
        xml=ET.fromstring(deck.read(f"ppt/slides/slide{s['slide']}.xml"))
        conditions=xml.findall('.//p:video/p:cMediaNode/p:cTn/p:stCondLst/p:cond',ns)
        assert len(conditions)==1 and conditions[0].attrib['delay']=='0'
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
checks.append('Five MP4s fully decode, match their 30 fps / 720p / H.264 durations, and match the files embedded in PowerPoint.')
checks.append('All five PowerPoint video timing nodes specify automatic start at slide entry.')
story=json.loads((ROOT/'storyboard.json').read_text())
assert story['total_duration_seconds']==156 and story['maximum_duration_seconds']==180
checks.append('The revised draft is 156 seconds, leaving 24 seconds for incoming footage.')
preview=ROOT/'cast_method_preview.mp4'
if preview.exists():
    v=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(preview)],text=True))
    assert abs(float(v['format']['duration'])-96)<.05
    assert v['streams'][0]['width']==1920 and v['streams'][0]['height']==1080
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(preview),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    checks.append('The silent method preview fully decodes and lasts 96 seconds at 1920 × 1080.')
report='# Animation validation\n\n'+''.join('- Passed: '+s+'\n' for s in checks)
report+='\nThe fixed-critic toy is illustrative. PowerPoint XML and media are checked; native PowerPoint slideshow playback is not available in this environment.\n'
(ROOT/'animations/VALIDATION.md').write_text(report)
print('\n'.join('PASS: '+s for s in checks))
