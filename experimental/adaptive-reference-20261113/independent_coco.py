"""Deterministic local COCO subset preparation. Never extracts arbitrary ZIP paths."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
import zipfile
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
ANNOTATION_SHA='113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ANNOTATION_BYTES=252907541
BUCKET='https://s3.amazonaws.com/images.cocodataset.org'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fetch(url,path,sha=None,size=None,maximum=20*1024*1024):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        if size is not None and path.stat().st_size!=size:raise ValueError('Existing download size mismatch')
        if sha is not None and digest(path)!=sha:raise ValueError('Existing download checksum mismatch')
        return
    partial=path.with_suffix(path.suffix+'.partial')
    try:
        total=0
        with urlopen(url,timeout=60) as response,partial.open('wb') as stream:
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>maximum:raise ValueError('Download exceeds declared size limit')
                stream.write(block)
                if total%(32*1024*1024)==0:print('downloaded MiB',total//1024//1024,flush=True)
        if size is not None and total!=size:raise ValueError('Incomplete download')
        if sha is not None and digest(partial)!=sha:raise ValueError('Download checksum mismatch')
        os.replace(partial,path)
    finally:partial.unlink(missing_ok=True)


def person_annotations(annotations):
    result=defaultdict(list)
    for a in annotations:
        if a['category_id']==1:result[a['image_id']].append(a)
    return result


def select_images(images,annotations,count=20):
    people=person_annotations(annotations)
    groups={name:[] for name in ('single','multiple','none')}
    for image in sorted(images,key=lambda x:x['id']):
        if image['license'] not in (4,5,7,8) or min(image['width'],image['height'])<240:continue
        persons=people[image['id']]
        if any(a.get('iscrowd') for a in persons):continue
        if any(not isinstance(a.get('segmentation'),list) or not a['segmentation'] for a in persons):continue
        fraction=sum(a['area'] for a in persons)/(image['width']*image['height'])
        if not persons:name='none'
        elif len(persons)==1 and .08<=fraction<=.7:name='single'
        elif 2<=len(persons)<=6 and .08<=fraction<=.7:name='multiple'
        else:continue
        if len(groups[name])<count:groups[name].append(image)
    if any(len(x)!=count for x in groups.values()):raise ValueError('Not enough eligible images; do not silently change protocol')
    return groups


def make_pairs(groups):
    pairs=[]
    for name,images in groups.items():
        if len(images)%2:raise ValueError('Pair count must be even')
        for i in range(0,len(images),2):
            pairs.append(dict(group=name,target=images[i]['id'],reference=images[i+1]['id'],
                              split='development' if i<4 else 'evaluation'))
    return pairs


def main():
    cache=ROOT/'.downloads/coco'
    out=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    out.mkdir(parents=True,exist_ok=True)
    archive=cache/'annotations_trainval2017.zip'
    fetch(BUCKET+'/annotations/annotations_trainval2017.zip',archive,ANNOTATION_SHA,ANNOTATION_BYTES,ANNOTATION_BYTES)
    with zipfile.ZipFile(archive) as z:
        with z.open('annotations/instances_val2017.json') as stream:data=json.load(stream)
    groups=select_images(data['images'],data['annotations'])
    people=person_annotations(data['annotations']);licenses={x['id']:x for x in data['licenses']}
    images=[x for entries in groups.values() for x in entries]
    def prepare(image):
        identifier=image['id'];name=image['file_name']
        if Path(name).name!=name:raise ValueError('Invalid dataset image path')
        source=cache/'images'/name
        fetch(BUCKET+'/val2017/'+name,source)
        rgb=Image.open(source).convert('RGB')
        if rgb.size!=(image['width'],image['height']):raise ValueError('Annotation/image dimension mismatch')
        mask=Image.new('L',rgb.size);draw=ImageDraw.Draw(mask)
        for annotation in people[identifier]:
            for polygon in annotation['segmentation']:
                if len(polygon)<6 or len(polygon)%2:raise ValueError('Invalid polygon')
                draw.polygon(list(zip(polygon[::2],polygon[1::2])),fill=255)
        rgb.thumbnail((640,640),Image.Resampling.BILINEAR)
        mask=mask.resize(rgb.size,Image.Resampling.NEAREST)
        inputpath=out/f'{identifier}-input.png';maskpath=out/f'{identifier}-human-mask.png'
        rgb.save(inputpath);mask.save(maskpath)
        return dict(id=identifier,file_name=name,coco_url=image.get('coco_url'),flickr_url=image.get('flickr_url'),
                    download_url=BUCKET+'/val2017/'+name,license=licenses[image['license']],
                    original_sha256=digest(source),input_sha256=digest(inputpath),mask_sha256=digest(maskpath),
                    person_count=len(people[identifier]),mask_fraction=float((np.asarray(mask)>0).mean()))
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(prepare,images))
    manifest=dict(annotation_sha256=ANNOTATION_SHA,protocol_sha256=digest(Path(__file__).with_name('independent-coco-protocol.md')),
                  groups={name:[x['id'] for x in entries] for name,entries in groups.items()},
                  pairs=make_pairs(groups),images=records,
                  scope='New project content; third-party model pretraining overlap unknown; evaluation not yet run')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(images=len(records),pairs=len(manifest['pairs']),development_pairs=6,evaluation_pairs=24),indent=2),flush=True)


if __name__=='__main__':main()
