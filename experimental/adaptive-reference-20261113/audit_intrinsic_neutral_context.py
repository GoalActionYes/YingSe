"""Audit automatic preprocessing, then reuse independent frozen-layer metrics."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import audit_intrinsic_intervention as common

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/adaptive-reference-20261113/intrinsic-neutral-context-v1'
SOURCE=common.SOURCE

def main():
    common.OUT=OUT;common.main()
    d=json.loads((OUT/'results.json').read_text());baseline=ROOT/'outputs/adaptive-reference-20261113/intrinsic-intervention-v1/results.json'
    assert common.digest(baseline)==d['control_results_sha256']
    control=json.loads(baseline.read_text());assert control['finished'] and control['config']==d['config'] and control['assets']==d['assets']
    checkpoint=ROOT/'.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    assert common.digest(checkpoint)==d['segmentation']['sha256']
    parent=json.loads((SOURCE/'results.json').read_text());rows=[];max_y=0.;files=0
    panel=Image.new('RGB',(1536,6*282),'white');draw=ImageDraw.Draw(panel)
    for ci,(case,prior) in enumerate(zip(d['cases'],parent['cases'],strict=True)):
        assert case['pair']==prior['pair'];person=np.array(Image.open(SOURCE/case['pair']/'reference-mask.png').convert('L'))>0
        identity_p=None;identity_image=None;tiles={}
        for row,old in zip(case['rows'],prior['variants'],strict=True):
            assert row['name']==old['name'];raw=np.array(Image.open(SOURCE/old['reference']['path']))
            context=row['prediction']['context'];p=OUT/context['probability_file'];assert common.digest(p)==context['probability_sha256']
            prob=np.load(p,allow_pickle=False);assert prob.shape==raw.shape[:2] and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            p=OUT/context['rgb_file'];assert common.digest(p)==context['rgb_sha256'];normalized=np.array(Image.open(p))
            x=raw.astype(float);gray=(x@np.array([.2126,.7152,.0722]))[...,None]
            expected=np.rint(x*prob[...,None].astype(float)+gray*(1-prob[...,None].astype(float))).clip(0,255).astype(np.uint8)
            assert np.array_equal(normalized,expected)
            ydelta=np.abs((normalized.astype(float)-x)@np.array([.2126,.7152,.0722]));max_y=max(max_y,float(ydelta.max()))
            assert ydelta.max()<=.500001
            if row['name']=='identity':identity_p=prob.copy();identity_image=normalized.copy()
            if row['name']=='identity_repeat':assert np.array_equal(identity_p,prob) and np.array_equal(identity_image,normalized)
            rows.append(dict(pair=case['pair'],name=row['name'],probability_sha256=context['probability_sha256'],normalized_sha256=context['rgb_sha256'],
                normalization_rgb_mae=float(np.abs(normalized.astype(float)-raw).mean()),
                person_normalization_rgb_mae=float(np.abs(normalized.astype(float)-raw)[person].mean()) if person.any() else None,
                automatic_person_fraction=float((prob>.5).mean())))
            if row['name'] in ['identity','background_green_100']:
                s=np.load(OUT/row['prediction']['file'],allow_pickle=False)[1].astype(float).clip(0,1)
                s=np.rint(np.where(s<=.0031308,s*12.92,1.055*s**(1/2.4)-.055)*255).astype(np.uint8)
                tiles[row['name']]=(raw,normalized,s)
            files+=2
        draw.text((4,ci*282+4),case['pair']+' | original / automatic neutral context / S | bg green / neutral context / S',fill='black')
        for j,a in enumerate(tiles['identity']+tiles['background_green_100']):panel.paste(Image.fromarray(a),(j*256,ci*282+26))
    assert files==144;panel.save(OUT/'context-panel.png')
    audit=json.loads((OUT/'audit.json').read_text());audit.update(context_files_checked=files,maximum_luminance_rounding_delta=max_y,
        context_repeats_exact=True,context_measurements=rows,
        note='Probability maps are hashed and preprocessing reconstructed; segmentation network not independently rerun')
    (OUT/'context-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['finished','passed','context_files_checked','maximum_luminance_rounding_delta','context_repeats_exact']}))

if __name__=='__main__':main()
