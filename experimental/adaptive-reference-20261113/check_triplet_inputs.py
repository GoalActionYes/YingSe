"""Verify downloaded pilot and exact decoded-pixel leakage before training."""
import hashlib
import json
import io
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageCms
from independent_coco import ROOT,digest


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/tst100k/cohort-v1'
    cohort=json.loads((out/'cohort.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'inputs-manifest.json').read_text(encoding='utf-8'))
    if digest(out/'cohort.json')!=manifest['cohort_sha256']:raise ValueError('Cohort changed')
    expected={r[role] for p in ['train','validation'] for r in cohort['groups'][p] for role in ['content','reference','gt']}
    if {e['path'] for e in manifest['files']}!=expected:raise ValueError('Incomplete or additional inputs')
    pixel_groups=defaultdict(set);file_groups=defaultdict(set);metadata={};color_errors=[]
    paths_to_split=defaultdict(set)
    for part in ['train','validation']:
        for row in cohort['groups'][part]:
            for role in ['content','reference','gt']:paths_to_split[row[role]].add(part)
    for entry in manifest['files']:
        path=out/'inputs'/entry['path']
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('Input hash mismatch')
        with Image.open(path) as im:
            mode=im.mode;size=list(im.size);orientation=im.getexif().get(274,1);profile_bytes=im.info.get('icc_profile');icc=bool(profile_bytes)
            bits=list(im.tag_v2.get(258,())) if hasattr(im,'tag_v2') else None
            profile_identity=None
            if profile_bytes:
                try:
                    profile=ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
                    profile_identity=(ImageCms.getProfileName(profile)+' '+ImageCms.getProfileDescription(profile)).strip()
                    if 'srgb' not in profile_identity.lower():color_errors.append(dict(path=entry['path'],reason='Embedded profile not identified as sRGB',profile=profile_identity))
                except Exception as exc:color_errors.append(dict(path=entry['path'],reason='Unreadable ICC',error=str(exc)))
            rgb=np.array(im.convert('RGB'))
        if mode!='RGB' or min(size)<16 or orientation not in (None,1):raise ValueError('Unsupported pilot mode/orientation')
        pixel_hash=hashlib.sha256(np.array(rgb.shape,dtype='<i8').tobytes()+rgb.tobytes()).hexdigest()
        metadata[entry['path']]=dict(mode=mode,size=size,orientation=orientation,icc_present=icc,profile_identity=profile_identity,
            tiff_bits_per_sample=bits,decoded_rgb_dtype=str(rgb.dtype),pixel_sha256=pixel_hash)
        pixel_groups[pixel_hash].update(paths_to_split[entry['path']]);file_groups[entry['sha256']].update(paths_to_split[entry['path']])
    invalid_pairs=[]
    for part in ['train','validation']:
        for row in cohort['groups'][part]:
            if metadata[row['content']]['size']!=metadata[row['gt']]['size']:invalid_pairs.append(dict(part=part,index=row['index'],reason='Content/GT size mismatch'))
    pixel_leaks=[h for h,v in pixel_groups.items() if len(v)>1];file_leaks=[h for h,v in file_groups.items() if len(v)>1]
    reserved_paths={r[role] for r in cohort['groups']['reserved'] for role in ['content','reference','gt']}
    unexpected_reserved=[p for p in reserved_paths if (out/'inputs'/p).exists()]
    passed=not (invalid_pairs or pixel_leaks or file_leaks or unexpected_reserved or color_errors)
    report=dict(scope='Integrity, exact duplicates and dimensions only; no perceptual alignment/subject leakage guarantee',
        source_sha256=digest(Path(__file__)),cohort_sha256=digest(out/'cohort.json'),manifest_sha256=digest(out/'inputs-manifest.json'),
        files=len(metadata),metadata=metadata,invalid_pairs=invalid_pairs,color_profile_errors=color_errors,cross_split_pixel_duplicates=pixel_leaks,
        cross_split_file_duplicates=file_leaks,reserved_paths_downloaded=unexpected_reserved,passed=passed,training_started=False,
        icc_policy='Embedded profile identity must identify sRGB; missing profile treated as dataset sRGB convention. No color conversion. PIL RGB decoding yields uint8, including quantization of 16-bit TIFF sources; not a 16-bit processing claim')
    (out/'quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='metadata'}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
