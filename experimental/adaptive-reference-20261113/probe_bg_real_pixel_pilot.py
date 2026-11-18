"""Verify subject fidelity and measure baseline sensitivity to photo backgrounds."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from background_intervention import region_delta, target_regions
from fetch_bg_real_pixel_pilot import cohort
from independent_coco import ROOT, digest
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import mask_for
from probe_reference_chroma_spread import PARENT, read
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


TARGETS = ('785-3553','9483-19432','872-1268','8021-8690')


def interior(mask):
    padded = np.pad(mask,5,constant_values=False)
    area = np.pad(padded,((1,0),(1,0))).cumsum(0).cumsum(1)
    return area[11:,11:]-area[:-11,11:]-area[11:,:-11]+area[:-11,:-11]==121


def measure(first,second,mask):
    delta = np.abs(first[mask].astype(float)-second[mask])
    return dict(pixels=int(mask.sum()),mae=float(delta.mean()) if delta.size else None,
                changed_pixels=int((delta>0).any(-1).sum()))


def run(inputs, out):
    evidence = ROOT/'docs/research/2026-11-18-bg-real-control-intake-evidence.json'
    meta = json.loads(evidence.read_bytes())
    selected = cohort(meta)
    manifest_file = inputs/'intake.json'
    manifest = json.loads(manifest_file.read_bytes())
    attribution = inputs/'attribution.json'
    assert set(json.loads(attribution.read_bytes())) == set(manifest) == {key for key,_ in selected}
    parent_file = PARENT/'results.json'
    assert digest(parent_file)==PARENT_SHA
    parent = json.loads(parent_file.read_bytes())
    targets = {row['pair']:row for row in parent['cases']}
    out.mkdir()
    data, ready = {}, {}
    for target_pair,(key,record) in zip(TARGETS,selected,strict=True):
        members = manifest[key]
        assert set(members)=={'C1','F1','mask'}
        saved = {}
        for role,member in members.items():
            filename = inputs/member['name']
            assert filename.parent==inputs and digest(filename)==member['sha256']
            if role=='mask':
                assert member['path']==record['subject_mask']
            else:
                assert member['sha256']==record['arms'][role]['sha256']
            saved[role] = filename
        c1,f1 = (decode_image(saved[role]) for role in ('C1','F1'))
        width,height = record['shape']
        assert c1.shape==f1.shape==(height,width,3)
        with Image.open(saved['mask']) as label:
            assert label.size==(width,height)
            mask = np.asarray(label.convert('L'))>=128
        core = interior(mask)
        outside = interior(~mask)
        stats = {name:measure(c1,f1,area) for name,area in
                 (('subject',mask),('subject_core',core),('subject_boundary',mask&~core),
                  ('background_core',outside),('background',~mask))}
        qualified = bool(core.sum()>=256 and outside.sum()>=256 and stats['subject_core']['mae']<=1
                         and stats['background_core']['mae']>=8)
        data[key] = dict(input=members, statistics=stats, qualified=qualified, target_pair=target_pair)
        if qualified:
            ready[key] = (c1,f1)

    checkpoint = None
    if len(ready)>=2:
        torch.set_num_threads(4)
        engine = SpatialEngine('multiref-v1')
        checkpoint = engine.checkpoint_sha
        assert checkpoint==parent['model_sha256']
        for key,pair in ready.items():
            row = targets[data[key]['target_pair']]
            target = read(row['input_file'])
            output_hashes = {}
            for role,reference in zip(('C1','F1'),pair,strict=True):
                photo = finish_spatial(target,engine.prepare(target,reference).apply(target),.8)
                path = out/f'{key}-{role}.png'
                Image.fromarray(photo).save(path)
                output_hashes[role] = digest(path)
            data[key]['outputs'] = output_hashes
        for key in ready:
            row = targets[data[key]['target_pair']]
            target = read(row['input_file'])
            person = mask_for(row,row['pair'].split('-')[0])
            masks = target_regions(target,person)
            masks['neutral_person'] = masks['neutral_proxy'] & person
            before, after = [decode_image(out/f'{key}-{role}.png') for role in ('C1','F1')]
            data[key]['output_changes'] = {name:region_delta(after,before,masks[name])
                                           for name in ('whole','person','neutral_person','neutral_proxy')}
    result = dict(cases=data, eligible=len(ready), checkpoint_sha256=checkpoint,
                  intake_sha256=digest(manifest_file), attribution_sha256=digest(attribution),
                  parent_sha256=PARENT_SHA, release_evidence_sha256=digest(evidence),
                  source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:[row['qualified'],row['statistics']['subject_core']['mae'],
                          row['statistics']['background_core']['mae']] for key,row in data.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    run(args.inputs,args.out)
