"""Throwaway automatic person/background reference-match probe, not product code.

Uses isolated copies of the existing feature/video plumbing. No source app is changed.
Run with experimental/skin-protection-20261019/runtime/python.exe -I -B this_file.
"""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'experimental/skin-protection-20261019'
OUT=ROOT/'outputs/region-match-probe';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(Path(__file__).parent),str(APP)]
os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache');os.environ['MPLBACKEND']='Agg'
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from reference_color_app import features,skin,media
from reference_color_app.lut import build_mapping,apply_mapping

# Experimental mask-aware statistics, deliberately separate from the base app.
source=(APP/'reference_color_app/features.py').read_text('utf8')
source=source.replace('from .contracts import FeaturePacket','from reference_color_app.contracts import FeaturePacket')
source=source.replace('def extract_features(rgb_u8, mode):','def extract_features(rgb_u8, mode, spatial_mask):')
source=source.replace('t,luma,endpoints,valid=histogram(q,mode)',
    "selected=cv2.resize(spatial_mask.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>.8\n if selected.sum()<64:raise ValueError('Insufficient reliable region pixels')\n t,luma,endpoints,valid=histogram(q[selected].reshape(-1,1,3),mode)")
source=source.replace('sample=q[::20,::20].reshape(-1,3);ys=sample[:,0];',
    'dense=q[selected];step=max(1,int(np.ceil(len(dense)/512)));sample=dense[::step];ys=sample[:,0];')
source='"""PROBE ONLY: masked histogram and at-most-512 regional moment samples; not native recovered semantics."""\n'+source
namespace={'cv2':cv2};exec(compile(source,'probe_masked_features.py','exec'),namespace)
(Path(__file__).parent/'probe_masked_features.py').write_text(source,'utf8')
extract_region=namespace['extract_features']


def segment(rgb):
    h,w=rgb.shape[:2];ratio=min(1,512/max(h,w));size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
    small=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    with skin._engine_lock:
        engine=skin._get_engine();mp=skin._mp
        result=engine.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small)))
        cat=result.category_mask.numpy_view().copy().squeeze()
        conf=sum(result.confidence_masks[k].numpy_view().copy().squeeze() for k in (1,2,3,4))
    # Hair + body skin + face skin + clothes. This is people, not arbitrary subjects.
    winning=np.isin(cat,(1,2,3,4));ramp=np.clip((conf-.45)/.4,0,1)
    mask=(ramp*ramp*(3-2*ramp)*winning).astype(np.float32)
    detected=int(np.count_nonzero(mask>.5))>=max(64,int(mask.size*.01))
    if not detected:mask.fill(0)
    mask=cv2.GaussianBlur(mask,(0,0),1.2)
    return mask,cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),dict(person_detected=bool(detected),person_fraction=float((mask>.5).mean()))


def fullmask(mask,rgb):return cv2.resize(mask,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)


def fit(target,reference):
    tm,_,td=segment(target);rm,_,rd=segment(reference)
    whole=build_mapping(features.extract_features(target,0),features.extract_features(reference,1))
    available=td['person_detected'] and rd['person_detected']
    result=dict(global_map=whole,person_map=None,background_map=None,enabled=False,source=td,reference=rd)
    if not available:
        result['fallback_reason']='Reference or target has no reliable person';return result
    masks=[fullmask(tm,target),fullmask(rm,reference)]
    for name,invert in [('person_map',False),('background_map',True)]:
        a,b=[1-m if invert else m for m in masks]
        if min(float((a>.8).mean()),float((b>.8).mean()))<.01:
            result['fallback_reason']='Insufficient confident foreground/background';return result
        result[name]=build_mapping(extract_region(target,0,a),extract_region(reference,1,b))
    result['enabled']=True;return result


class RegionalSession:
    mapping=None
    def __init__(self):self.history=None
    def process(self,original,unused,temporal=False):
        m,gray,info=segment(original);pack=type(self).mapping
        if not pack['enabled']:
            self.history=None
            return apply_mapping(original,pack['global_map'],.8),dict(skin_detected=False,**info,fallback_reason=pack['fallback_reason'])
        if not info['person_detected']:
            self.history=None
            return apply_mapping(original,pack['global_map'],.8),dict(skin_detected=False,**info,fallback_reason='No reliable person in frame')
        temporal_state='static'
        if temporal and self.history is not None and self.history[0].shape==gray.shape:
            prev,pm=self.history
            difference=float(np.abs(gray.astype(np.float32)-prev).mean())/255
            if difference<.15:
                flow=cv2.calcOpticalFlowFarneback(gray,prev,None,.5,3,15,3,5,1.2,0)
                yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]].astype(np.float32)
                mx,my=xx+flow[...,0],yy+flow[...,1]
                warped=cv2.remap(pm,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                oldgray=cv2.remap(prev,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                trust=(np.abs(gray.astype(np.float32)-oldgray)<20)&(mx>=0)&(mx<gray.shape[1]-1)&(my>=0)&(my<gray.shape[0]-1)
                m=np.where(trust,.8*m+.2*warped,m).astype(np.float32);temporal_state='motion_aligned'
            else:temporal_state='cut_reset'
        self.history=(gray.copy(),m.copy()) if temporal else None
        weight=fullmask(m,original)[...,None]
        person=apply_mapping(original,pack['person_map'],.8)
        background=apply_mapping(original,pack['background_map'],.8)
        output=np.rint(weight*person+(1-weight)*background).clip(0,255).astype(np.uint8)
        return output,dict(skin_detected=True,**info,temporal_state=temporal_state,
                          algorithm='experimental masked feature statistics, two fixed maps, moving person alpha')


def save(name,rgb):Image.fromarray(rgb).save(OUT/name)


def sheet(name,panels):
    tw,th=480,300;canvas=Image.new('RGB',(tw*2,(th+35)*int(np.ceil(len(panels)/2))),'#161d23')
    draw=ImageDraw.Draw(canvas)
    for i,(label,rgb) in enumerate(panels):
        im=Image.fromarray(rgb);im.thumbnail((tw,th));x=(i%2)*tw;y=(i//2)*(th+35)
        draw.text((x+12,y+10),label,fill='white');canvas.paste(im,(x+(tw-im.width)//2,y+35+(th-im.height)//2))
    canvas.save(OUT/name)


def main():
    begin=time.monotonic();video=ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
    target,_=media.decode_frame(video,0);mask,_,person=segment(target);a=fullmask(mask,target)[...,None]
    save('target.png',target);save('person-mask.png',np.repeat(np.rint(a*255).astype(np.uint8),3,axis=2))
    # Controlled reference: same composition, different known foreground/background tints.
    f=target.astype(np.float32)
    person_ref=np.clip(f*np.array([1.02,.96,.93])+[5,0,0],0,255)
    bg_ref=np.clip(f*np.array([.87,.99,1.07])+[-3,2,7],0,255)
    reference=np.rint(a*person_ref+(1-a)*bg_ref).astype(np.uint8);save('reference-controlled.png',reference)
    pack=fit(target,reference);assert pack['enabled'],pack
    RegionalSession.mapping=pack
    global_result=apply_mapping(target,pack['global_map'],.8)
    region_result,diag=RegionalSession().process(target,target)
    save('global-match.png',global_result);save('regional-match.png',region_result)
    weight=fullmask(mask,target)
    metrics={}
    for label,selection in [('person',weight>.9),('background',weight<.1)]:
        metrics[label]={k:float(np.abs(v.astype(float)-reference)[selection].mean()) for k,v in [('global_mae',global_result),('regional_mae',region_result)]}
    sheet('image-comparison.png',[('Target',target),('Controlled reference',reference),('Global match 80%',global_result),('Person/background match 80%',region_result)])
    # Diagnostics / validation, not an accuracy benchmark.
    np.testing.assert_array_equal(apply_mapping(target,pack['global_map'],0),target)
    no_person=np.zeros((128,128,3),np.uint8);no_person[:]=[60,170,220]
    fallback=fit(target,no_person);assert not fallback['enabled']
    # Region stats must ignore unselected pixels in this <=680px test.
    selected=weight>.9;changed=target.copy();changed[~selected]=[255,0,255]
    np.testing.assert_array_equal(extract_region(target,0,selected.astype(np.float32)).values,
                                  extract_region(changed,0,selected.astype(np.float32)).values)
    # Reuse a private copy of the verified codec lifecycle, replacing only the pixel processor.
    code=(APP/'reference_color_app/protected_video.py').read_text('utf8')
    code=code.replace('from . import media','from reference_color_app import media')
    code=code.replace('from .lut import apply_mapping, _strength','from reference_color_app.lut import apply_mapping, _strength')
    code=code.replace('from .skin import SkinSession','from __main__ import RegionalSession as SkinSession')
    code=code.replace('graded = apply_mapping(original, mapping, strength)','graded = original  # regional processor applies the two experimental maps')
    code='"""PROBE COPY: regional pixel processor, not shipped protection pipeline."""\n'+code
    (Path(__file__).parent/'probe_video.py').write_text(code,'utf8')
    scope={};exec(compile(code,'probe_video.py','exec'),scope)
    started=time.monotonic();children=[]
    settings=scope['export_protected_video'](video,OUT/'regional-video.mp4',pack['global_map'],.8,OUT/'video-job',lambda:False,
        lambda x:None,lambda processes:children.extend(p for p in processes if p not in children))
    assert settings['fps']==30 and settings['frames']==settings['verified_frames']==90
    assert all(p.poll() is not None for p in children)
    sample_panels=[]
    for t in (0,1,2.5):
        orig,_=media.decode_frame(video,t);result,_=media.decode_frame(OUT/'regional-video.mp4',t)
        sample_panels.extend([(f'Input {t}s',orig),(f'Regional encoded {t}s',result)])
    sheet('video-contact-sheet.png',sample_panels)
    # Different-content reference: exposes scene/material influence instead of hiding it.
    other=media.decode_image(APP/'reference_color_app/demo/skin-reference.png')
    alternative=fit(target,other);RegionalSession.mapping=alternative
    alternative_result,alternative_diag=RegionalSession().process(target,target)
    save('different-content-reference.png',other);save('different-content-result.png',alternative_result)
    sheet('different-content-comparison.png',[('Target',target),('Different-content reference',other),
        ('Global match',apply_mapping(target,alternative['global_map'],.8)),('Regional match',alternative_result)])
    report=dict(probe_only=True,product_ui_added=False,inputs=dict(video=str(video.relative_to(ROOT)),reference='controlled same-content tint; plus official different-content reference'),
        model_sha256=skin.MODEL_SHA256,person_classes=[1,2,3,4],person_info=person,reference_info=pack['reference'],
        feature_policy='Masked histogram and <=512 deterministic regional samples; modified statistics, not native semantics',
        strength=.8,controlled_reference_mae=metrics,video=settings,video_elapsed_seconds=time.monotonic()-started,
        assertions=dict(zero_strength_exact=True,no_reference_person_explicit_fallback=True,regional_statistics_ignore_unselected_pixels=True,
                        video_frames_verified=True,all_child_processes_exited=True),
        limitations=['Single official person clip, composited background','No multi-shot, broad skin tones or fast occlusion acceptance',
                     'Person region includes hair and clothes; reference clothing colors can bias the map','Whole background is one region, sky/trees not distinguished',
                     'Fixed maps from first target frame; moving masks, no automatic shot-wise reanalysis','No new app UI or distribution created'],
        total_seconds=time.monotonic()-begin)
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps(dict(done=True,frames=settings['frames'],mae=metrics,total_seconds=report['total_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
