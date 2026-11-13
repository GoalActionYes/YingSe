"""Read only current 64 development predictions and fixed grouped GTs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from tone_response import tone_response


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='tone-response-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected directory name')
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';model=base/'multiref-control-v1';epoch=model/'epoch-6'
    checkpoint_sha='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    if digest(model/'epoch-6.pth')!=checkpoint_sha:raise ValueError('Current checkpoint changed')
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));prep=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'));run=json.loads((model/'results.json').read_text(encoding='utf-8'))
    scores=json.loads((epoch/'scores.json').read_text(encoding='utf-8'));lookup={r['original_path']:r for r in prep['files']};predictions={r['index']:r for r in scores['cases']}
    if not prep['passed'] or prep['cohort_sha256']!=digest(data/'cohort.json') or run['cohort_sha256']!=prep['cohort_sha256'] or run['prepared_manifest_sha256']!=digest(data/'srgb-v1/manifest.json') or len(predictions)!=64:raise ValueError('Changed data or scores')
    groups=defaultdict(dict)
    for row in cohort['groups']['validation']:
        if row['style_index'] in groups[row['group_id']]:raise ValueError('Duplicate style')
        groups[row['group_id']][row['style_index']]=row
    if len(groups)!=32 or any(set(v)!={0,1} for v in groups.values()):raise ValueError('Incomplete development groups')
    out=base/args.run;out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['diagnose_multiref_tones.py','tone_response.py','tone-response-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Descriptive already-consumed development diagnostics; no independent/semantic/perceptual acceptance',checkpoint_sha256=checkpoint_sha,
        cohort_sha256=prep['cohort_sha256'],prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),scores_sha256=digest(epoch/'scores.json'),
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,rows in groups.items():
        images=[]
        for style in [0,1]:
            row=rows[style];pred=epoch/(str(row['index'])+'.png');entry=lookup[row['gt']];gt=data/entry['prepared_path']
            if digest(pred)!=predictions[row['index']]['sha256'] or digest(gt)!=entry['sha256']:raise ValueError('Changed prediction or GT')
            pair=[]
            for path in [pred,gt]:
                with Image.open(path) as image:pair.append(np.array(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
            images.append(pair)
        response=tone_response(images[0][0],images[1][0],images[0][1],images[1][1]);report['cases'].append(dict(group_id=name,indices=[rows[i]['index'] for i in [0,1]],**response))
    keys=report['cases'][0]['prediction_delta'];report['summary']={}
    for key in keys:
        active=[r for r in report['cases'] if r['direction_correct'][key] is not None]
        report['summary'][key]=dict(mean_absolute_delta_error=float(np.mean([r['absolute_delta_error'][key] for r in report['cases']])),
            mean_absolute_gt_delta=float(np.mean([abs(r['gt_delta'][key]) for r in report['cases']])),
            mean_absolute_prediction_delta=float(np.mean([abs(r['prediction_delta'][key]) for r in report['cases']])),
            significant_groups=len(active),direction_correct=sum(r['direction_correct'][key] for r in active))
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
