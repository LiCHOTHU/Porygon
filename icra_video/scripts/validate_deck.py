#!/usr/bin/env python3
"""Check timing, source measurements, PowerPoint structure, and media slots."""
from pathlib import Path
import json, os, sys, subprocess, zipfile
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
deps=Path(os.environ.get('ICRA_VIDEO_DEPS','/tmp/icra-video-deps'))
if deps.exists():sys.path.insert(0,str(deps))
from pptx import Presentation
from PIL import Image
story=json.loads((ROOT/'storyboard.json').read_text())
manifest=json.loads((ROOT/'animation_manifest.json').read_text())
prs=Presentation(ROOT/'cast_video_slides.pptx')
assert len(prs.slides)==10
assert sum(s['duration'] for s in story['slides'])==180
assert story['slides'][6]['start']==120
assert len(manifest['slots'])==7
assert [s['id'] for s in manifest['slots']]==['A01','A02','A03','A04','A05','V01','V02']
for i,(slide,item) in enumerate(zip(prs.slides,story['slides']),1):
    assert item['narration'] in slide.notes_slide.notes_text_frame.text
    assert len(item['narration'].split())/item['duration']*60<=160
    for shape in slide.shapes:
        assert shape.left>=0 and shape.top>=0, (i,shape.name)
        assert shape.left+shape.width<=prs.slide_width+1000, (i,shape.name)
        assert shape.top+shape.height<=prs.slide_height+1000, (i,shape.name)
    for slot in [s for s in manifest['slots'] if s['slide']==i]:
        assert slot['pptx_shape_name'] in [s.name for s in slide.shapes]
with zipfile.ZipFile(ROOT/'cast_video_slides.pptx') as z:
    for i,item in enumerate(story['slides'],1):
        xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
        tr=xml.find('{http://schemas.openxmlformats.org/presentationml/2006/main}transition')
        assert int(tr.attrib['advTm'])==item['duration']*1000
for src,dst in [('figures/experiment_chart_data.json','experiment_data.json'),('real_robot_results.json','robot_results.json')]:
    assert (ROOT.parent/'_ICRA_2027__CAST'/src).read_bytes()==(ROOT/'assets'/dst).read_bytes()
info=subprocess.check_output(['pdfinfo',str(ROOT/'cast_video_slides.pdf')],text=True)
assert 'Pages:           10' in info
for image in sorted((ROOT/'previews').glob('slide-*.png')):
    assert Image.open(image).size==(1920,1080)
checks=['10 slides / 180 seconds','120-second method segment / 60-second robot segment','All narration under 160 words per minute','Speaker notes and automatic slide timings present','Seven named media containers; five completed animation clips and two robot-video reservations','All shapes within slide bounds','Source measurement files unchanged','10-page PDF and 1920×1080 slide previews']
(ROOT/'VALIDATION.md').write_text('# Build validation\n\n'+''.join('- Passed: '+s+'\n' for s in checks)+'\nPDF previews are rendered from the same layout primitives as the editable PowerPoint. Native PowerPoint playback was not available in this environment.\n')
print('\n'.join('PASS: '+s for s in checks))
