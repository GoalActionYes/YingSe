"""V2 control semantics and independent FFmpeg rendering checks."""
from pathlib import Path
import json, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import grading as g
import server as s

ROOT=Path(__file__).resolve().parent
OUT=s.OUT/'style-check'; OUT.mkdir(exist_ok=True)
checks=[]
def check(name,ok,**details):
    checks.append(dict(name=name,passed=bool(ok),**details))
    print(('PASS ' if ok else 'FAIL ')+name,flush=True)

def one(i):
    w=[0]*9;w[i]=1;return w

source=g.load_image(s.DEMO/'source.png',640)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
def contact(path,filename):
    src=g.load_image(path,700)
    page=Image.new('RGB',(1080,990),'#111813');draw=ImageDraw.Draw(page)
    for i,label in enumerate(g.LABELS):
        result=g.apply_lut(src,g.make_lut(one(i)))
        im=Image.fromarray(np.round(result*255).astype('uint8'));im.thumbnail((348,280))
        x=(i%3)*360;y=(i//3)*330
        page.paste(im,(x+(360-im.width)//2,y+40+(280-im.height)//2))
        draw.text((x+12,y+7),label,font=font,fill='#d3eab9')
    page.save(OUT/filename)

results=[]
for i in range(1,9):
    weights=one(i)
    full=g.apply_lut(source,g.make_lut(weights,1))
    half=g.apply_lut(source,g.make_lut(weights,.5))
    zero=g.apply_lut(source,g.make_lut(weights,0))
    err0=float(np.max(abs(zero-source)))
    err50=float(np.max(abs(half-(source+full)/2)))
    check(g.STYLE_IDS[i]+'_strength_0_50_100',err0<1e-6 and err50<1e-6,zero_error=err0,half_error=err50)
    check(g.STYLE_IDS[i]+'_visible_change',np.mean(abs(full-source))>.025,mean_rgb_change=float(np.mean(abs(full-source))))
    results.append(full)
    # Test the actual exported file through a separate renderer, including non-neutral controls.
    a={'exposure':.2,'contrast':17,'saturation':-12,'temperature':25}
    lut=g.make_lut(weights,.8,adjustments=a)
    (OUT/'check.cube').write_text(g.cube_text(lut),encoding='ascii')
    original=g.load_image(s.DEMO/'source.png')
    expected=g.apply_lut(original,lut)
    p=s.run_ff(['-y','-i',str(s.DEMO/'source.png'),'-vf','lut3d=file=check.cube:interp=trilinear',
                '-frames:v','1','-threads','1',str(OUT/'ffmpeg.png')],cwd=OUT,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf8','replace'))
    actual=g.load_image(OUT/'ffmpeg.png')
    error=np.abs(np.round(expected*255)-np.round(actual*255))
    check(g.STYLE_IDS[i]+'_export_consistency',error.max()<=2,mean_8bit_error=float(error.mean()),max_8bit_error=float(error.max()))

distances=[float(np.mean(abs(a-b))) for i,a in enumerate(results) for b in results[i+1:]]
check('eight_distinct_looks',min(distances)>.02,minimum_pairwise_rgb_difference=min(distances))
mono=g.apply_lut(source,g.make_lut(one(0),adjustments={'saturation':-100}))
check('saturation_minus100_removes_color',np.max(abs(mono[...,0]-mono[...,2]))<1e-6)
for key,(low,high) in g.CONTROL_LIMITS.items():
    for value in (low,high):
        out=g.apply_lut(source,g.make_lut(one(0),0,adjustments={key:value}))
        delta=float(np.mean(abs(out-source)))
        check(f'{key}_{value}_independent_of_filter',delta>.02,mean_rgb_change=delta)
w=[.35,.1,.1,.2,.05,.05,.05,.05,.05]
mix,strength=g.split_strength(w)
a=g.apply_lut(source,g.make_lut(w))
b=g.apply_lut(source,g.make_lut(mix,strength))
check('recommended_identity_becomes_strength',np.max(abs(a-b))<1e-6 and mix[0]==0,unfiltered_fraction=.35,strength=strength)

contact(s.DEMO/'source.png','styles-calibration.jpg')
photo=ROOT/'data'/'36c37fca589a477fa7c06f50d783a549.jpg'
if photo.exists():contact(photo,'styles-portrait.jpg')
night=ROOT/'data'/'c7e4ac67a15c4eb5837040267da643ee.jpg'
if night.exists():contact(night,'styles-blue-light.jpg')
page=Image.new('RGB',(1080,460),'#111813');d=ImageDraw.Draw(page)
im=g.load_image(photo if photo.exists() else s.DEMO/'source.png',600)
for i,strength in enumerate([0,.5,1]):
    out=g.apply_lut(im,g.make_lut(one(3),strength))
    thumb=Image.fromarray(np.round(out*255).astype('uint8'));thumb.thumbnail((348,390))
    page.paste(thumb,(i*360+(360-thumb.width)//2,50));d.text((i*360+12,10),f'青橙 · {int(strength*100)}%',font=font,fill='#d3eab9')
page.save(OUT/'strength-0-50-100.jpg')
report={'palette_version':g.PALETTE_VERSION,'created_local':time.strftime('%Y-%m-%d %H:%M:%S'),
        'all_passed':all(c['passed'] for c in checks),'checks':checks,
        'scope':'Color/control semantics and FFmpeg agreement; does not establish subjective quality or semantic skin protection.'}
(ROOT/'style-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(f'{len(checks)} checks; all_passed={report["all_passed"]}')
if not report['all_passed']:raise SystemExit(1)
