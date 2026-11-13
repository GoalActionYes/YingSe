"""Make verified ICC-managed copies while preserving every original byte."""
import json
import argparse
from collections import defaultdict
import hashlib
from PIL import Image
import numpy as np
from independent_coco import ROOT,digest
from triplet_color import to_srgb
from triplet_dataset_options import dataset_options


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--expanded',action='store_true');mode.add_argument('--multiref',action='store_true');args=parser.parse_args()
    base=dataset_options(ROOT,args.expanded,args.multiref)['directory']
    cohort=json.loads((base/'cohort.json').read_text(encoding='utf-8'));manifest=json.loads((base/'inputs-manifest.json').read_text(encoding='utf-8'))
    quality=json.loads((base/'quality.json').read_text(encoding='utf-8'))
    if any(quality[k] for k in ['invalid_pairs','cross_split_pixel_duplicates','cross_split_file_duplicates','reserved_paths_downloaded']):raise ValueError('Non-color data issue unresolved')
    if digest(base/'cohort.json')!=manifest['cohort_sha256'] or digest(base/'inputs-manifest.json')!=quality['manifest_sha256']:raise ValueError('Input manifest changed')
    out=base/'srgb-v1';out.mkdir(exist_ok=False);records=[];pixel_splits=defaultdict(set)
    membership=defaultdict(set)
    for part in ['train','validation']:
        for row in cohort['groups'][part]:
            for role in ['content','reference','gt']:membership[row[role]].add(part)
    for entry in manifest['files']:
        original=base/'inputs'/entry['path']
        if digest(original)!=entry['sha256']:raise ValueError('Original input changed')
        with Image.open(original) as image:prepared,info=to_srgb(image)
        path=out/(entry['path']+'.png');path.parent.mkdir(parents=True,exist_ok=True);prepared.save(path,icc_profile=prepared.info['icc_profile'])
        with Image.open(path) as check:
            rgb=np.array(check);same=np.array_equal(rgb,np.array(prepared))
            if check.mode!='RGB' or not check.info.get('icc_profile') or not same:raise ValueError('Prepared RGB did not round-trip')
        if digest(original)!=entry['sha256']:raise ValueError('Preparation changed source')
        pixel_hash=hashlib.sha256(np.array(rgb.shape,dtype='<i8').tobytes()+rgb.tobytes()).hexdigest();pixel_splits[pixel_hash].update(membership[entry['path']])
        records.append(dict(original_path=entry['path'],original_sha256=entry['sha256'],prepared_path=str(path.relative_to(base)),sha256=digest(path),pixel_sha256=pixel_hash,preparation=info))
    leaks=[h for h,p in pixel_splits.items() if len(p)>1]
    report=dict(cohort_sha256=digest(base/'cohort.json'),inputs_manifest_sha256=digest(base/'inputs-manifest.json'),
        source_sha256={name:digest(ROOT/'experimental/adaptive-reference-20261113'/name) for name in ['triplet_color.py','prepare_triplet_srgb.py']},
        files=records,converted_files=len(records),cross_split_prepared_pixel_duplicates=leaks,passed=not leaks,
        scope='RGB8 ICC-managed academic pilot; no 16-bit preservation or perceptual alignment guarantee',training_started=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(records),leaks=len(leaks),passed=not leaks)),flush=True)
    if leaks:raise SystemExit(1)


if __name__=='__main__':main()
