"""Recompute all expert-label and soft-region differences without measurement helpers."""
import io
import json
from pathlib import Path
import zlib
import numpy as np
from PIL import Image,ImageCms
from independent_coco import ROOT,digest

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-expert-contrast-v1'
OLD=BASE/'ppr-pixel-sample-v1'


def main():
    d=json.loads((OUT/'measurement.json').read_text());r=json.loads((OUT/'results.json').read_text());p=json.loads((OUT/'plan.json').read_text());old=json.loads((OLD/'results.json').read_text())
    assert d['finished'] and r['finished'] and not d['training_started']
    assert digest(OUT/'results.json')==d['acquisition_sha256']
    assert digest(Path(__file__).with_name('measure_ppr_expert_contrast.py'))==d['driver_sha256']
    all_files=[dict(x,expert='a') for x in old['files'] if x['role']=='target_a']+r['files']
    arrays={};small={}
    for row in all_files:
        path=ROOT/row['local_path'];raw=path.read_bytes()
        assert digest(path)==row['sha256'] and len(raw)==row['bytes'] and zlib.crc32(raw)==row['crc']
        with Image.open(path) as im:
            assert tuple(im.tag_v2[258])==(8,8,8)
            profile=im.info.get('icc_profile')
            rgb=im.convert('RGB')
            if profile:rgb=ImageCms.profileToProfile(rgb,ImageCms.ImageCmsProfile(io.BytesIO(profile)),ImageCms.createProfile('sRGB'),outputMode='RGB',renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
            arrays[row['expert'],row['id']]=np.asarray(rgb).astype(np.int64)
            small[row['expert'],row['id']]=np.asarray(rgb.resize((256,256),Image.Resampling.BILINEAR)).astype(np.int64)
    maximum=0.
    for item in d['cases']:
        key=item['id'];a,b=item['pair'].split('_')
        for name,lookup in [('native',arrays),('model_256',small)]:
            delta=lookup[b,key]-lookup[a,key];absolute=np.abs(delta);expected=item[name]
            assert expected['exact']==bool((delta==0).all()) and expected['max_difference']==int(absolute.max())
            differences=[abs(float(absolute.sum()/absolute.size)-expected['mae'])]
            differences.extend(abs(delta.reshape(-1,3).sum(axis=0)/delta.shape[0]/delta.shape[1]-np.array(expected['mean_rgb_second_minus_first'])))
            maximum=max(maximum,*differences)
        maskrow=next(x for x in old['files'] if x['role']=='mask' and x['id']==key)
        path=ROOT/maskrow['local_path'];assert digest(path)==maskrow['sha256']
        with Image.open(path) as im:alpha=np.asarray(im,dtype=np.float64)/255
        delta=arrays[b,key]-arrays[a,key]
        for region,weight in [('human',alpha),('background',1-alpha)]:
            values=item['offline_soft_mask_regions'][region]
            mean=np.einsum('hwc,hw->c',delta,weight)/np.sum(weight)
            mae=np.einsum('hwc,hw->',np.abs(delta),weight)/(np.sum(weight)*3)
            maximum=max(maximum,float(np.abs(mean-values['mean_rgb_second_minus_first']).max()),abs(float(mae)-values['mae']))
    assert len(d['cases'])==12 and maximum<1e-9
    assert {x['id'] for x in d['cases']}==set(p['ids'])
    audit=dict(finished=True,measurement_sha256=digest(OUT/'measurement.json'),driver_sha256=digest(__file__),files_crc_sha_verified=12,label_pairs_recomputed=12,max_numeric_difference=maximum,source_icc_policy_reimplemented=True,shared_pixel_decoder='Pillow/LCMS; not a fully separate codec',training_started=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))


if __name__=='__main__':main()
