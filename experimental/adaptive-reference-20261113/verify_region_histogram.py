"""Post-prediction producer/cache audit. Never used to tune the candidate."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
import torch
from independent_coco import ROOT,digest
from controlled_styles import stylize
from person_segmenter import PersonSegmenter,CHECKPOINT_SHA
from spatial_render import SpatialEngine
from region_histogram import weighted_histogram


def rgb(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))


def main():
    torch.set_num_threads(4)
    data=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    out=ROOT/'outputs/adaptive-reference-20261113/region-histogram-20261115-v1'
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'));lookup={r['id']:r for r in manifest['images']}
    # Guard PersonSegmenter's optional downloader before constructing it.
    local=ROOT/'.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    if not local.is_file() or digest(local)!=CHECKPOINT_SHA:raise ValueError('Missing local frozen segmentation weights; no download allowed')
    engine=SpatialEngine('multiref-v1');segmenter=PersonSegmenter();segrows={};rows=[];histdiff=[]
    for row in report['cases']:
        job=row['job'];target=rgb(data/f"{job['target']}-input.png");reference=rgb(data/job['reference_file'])
        original_reference=data/f"{job['reference']}-input.png"
        if digest(original_reference)!=lookup[job['reference']]['input_sha256']:raise ValueError('Reference original changed')
        for role,source,expected in [('target',target,rgb(data/job['truth_file'])),('reference',rgb(original_reference),reference)]:
            maskfile=data/f"{job[role]}-human-mask.png"
            if digest(maskfile)!=lookup[job[role]]['mask_sha256']:raise ValueError('Annotation changed')
            with Image.open(maskfile) as maskimage:mask=np.asarray(maskimage.filter(ImageFilter.GaussianBlur(2)),dtype=float)/255
            if not np.array_equal(stylize(source,mask,job['profile']),expected):raise ValueError('Controlled image does not match original producer')
        for name,image in [(f"{job['target']}-target",target),(f"{job['reference']}-{job['profile']}",reference)]:
            if name in segrows:continue
            cached=np.load(data/'development/automatic-masks'/f'{name}.npy',allow_pickle=False)
            # Original cache producer did not disable cuDNN's default TF32.
            # SpatialEngine does disable it; restore the producer's setting
            # only around this provenance check, never around color inference.
            previous=torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32=True
                actual,timing=segmenter.predict(image)
            finally:torch.backends.cudnn.allow_tf32=previous
            delta=float(np.abs(actual-cached).max())
            segrows[name]=dict(maximum_probability_difference=delta,seconds=timing['seconds'])
            if delta>1e-5:raise ValueError('Cached segmentation cannot be reproduced within 1e-5')
        with torch.inference_mode():
            c=engine.analysis(target);y=engine.api['srgb_to_ycbcr'](c)
            weighted=weighted_histogram(y,torch.ones_like(y[:,:1]));author=engine.api['batch_histogram_1D'](y)
            difference=float((weighted-author).abs().max());histdiff.append(difference)
            if difference>1e-7:raise ValueError('All-one histogram changed author definition')
        baseline=engine.prepare(target,reference).apply(target,1)
        entry=row['outputs']['baseline'];path=out/entry['file']
        if digest(path)!=entry['sha256'] or not np.array_equal(baseline,rgb(path)):raise ValueError('Baseline changed')
        for value in row['outputs'].values():
            if digest(out/value['file'])!=value['sha256']:raise ValueError('Prediction hash changed')
        rows.append(dict(index=row['index'],controlled_reference_exact=True,controlled_truth_exact=True,baseline_pixel_exact=True))
    if any(digest(Path(__file__).with_name(n))!=v for n,v in report['source_sha256'].items()):raise ValueError('Frozen producer changed')
    audit=dict(passed=True,results_sha256=digest(out/'results.json'),verifier_sha256=digest(Path(__file__)),cases=rows,segmentation=segrows,maximum_all_one_histogram_difference=max(histdiff),segmentation_maximum_difference=max(r['maximum_probability_difference'] for r in segrows.values()),segmentation_cudnn_tf32=True,color_inference_cudnn_tf32=False,source_and_weights_unchanged=True,scope='Post-prediction input/producer/cache/baseline audit, not a new candidate or human acceptance')
    (out/'producer-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=len(rows),segmentation_inputs=len(segrows),maximum_histogram_difference=max(histdiff),maximum_segmentation_difference=audit['segmentation_maximum_difference'])),flush=True)


if __name__=='__main__':main()
