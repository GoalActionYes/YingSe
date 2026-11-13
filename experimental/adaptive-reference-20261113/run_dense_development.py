"""Fixed development6-15: freeze inputs/methods, predict all, only then fetch expert targets."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from canonical_model import ROOT,CanonicalModel
from independent_coco import fetch,digest
from dense_anchor_model import DenseAnchorModel
from dense_anchors import match_anchors,color_from_anchors,choose_blend
from check_dense_anchors import tensor,rgb8
from prepare_salut_assets import DATA
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze,algorithm_fingerprint
from reference_color_app.lut import apply_mapping

INDICES=tuple(range(6,16))


def load(path):
    with Image.open(path) as image: return np.array(image.convert('RGB'))


def working(rgb):
    image=Image.fromarray(rgb); image.thumbnail((640,640),Image.Resampling.BILINEAR); return np.array(image)


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/pst50/development-dense-round2-v1'
    out.mkdir(exist_ok=False)
    names=('dense_anchors.py','dense_anchor_model.py','run_dense_development.py','check_dense_anchors.py',
        'dense-anchor-protocol.md','dense-development-round2-protocol.md','prepare_dense_anchor.py','canonical_model.py')
    frozen=out/'frozen-source'; frozen.mkdir()
    hashes={}
    for name in names:
        source=Path(__file__).with_name(name); (frozen/name).write_bytes(source.read_bytes()); hashes[name]=digest(source)
    cohort=dict(scope='Development round2, no unseen-test or product acceptance claim',development_indices=list(INDICES),
        previously_used_indices=list(range(1,6)),still_reserved_indices=list(range(16,51)),dataset_revision=DATA,
        source_sha256=hashes,current_algorithm_fingerprint=algorithm_fingerprint(),
        previous_trial_sha256=digest(out.parent/'development-dense-anchors-native-v1/results.json'),
        expert_targets_not_downloaded_at_freeze=True)
    (out/'cohort.json').write_text(json.dumps(cohort,indent=2),encoding='utf-8')
    entries=[]
    for folder,prefix in [('content_709','in'),('paired_style','tar'),('paired_gt','gt')]:
        with urlopen(f'https://huggingface.co/api/datasets/zrgong/PST50/tree/{DATA}/{folder}',timeout=30) as response:
            listing=json.load(response)
        wanted={f'{folder}/{prefix}{i}.png' for i in INDICES}
        found=[x for x in listing if x['path'] in wanted]
        if len(found)!=len(INDICES): raise ValueError('Unexpected paired asset set')
        entries.extend(found)
    def prepare(entry):
        path=out/'inputs'/entry['path']; expected=entry['lfs']['oid']; size=entry['size']
        fetch(f"https://huggingface.co/datasets/zrgong/PST50/resolve/{DATA}/{quote(entry['path'])}",path,expected,size,30*1024*1024)
        return dict(path=str(path.relative_to(out)),sha256=digest(path),bytes=size)
    source_entries=[x for x in entries if not x['path'].startswith('paired_gt/')]
    with ThreadPoolExecutor(max_workers=3) as pool: prepared=list(pool.map(prepare,source_entries))
    (out/'inputs-manifest.json').write_text(json.dumps(prepared,indent=2),encoding='utf-8')
    torch.set_num_threads(4); canonical=CanonicalModel(); model=DenseAnchorModel()
    prediction_dir=out/'predictions'; prediction_dir.mkdir()
    cases=[]
    report=dict(cohort_sha256=digest(out/'cohort.json'),descriptor=model.provenance,canonical=canonical.provenance,cases=cases)
    def save(): (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    for index in INDICES:
        row=dict(index=index,outputs={},expert_target_read=False); cases.append(row)
        try:
            target_path=out/f'inputs/content_709/in{index}.png'; reference_path=out/f'inputs/paired_style/tar{index}.png'
            target,reference=load(target_path),load(reference_path)
            start=time.perf_counter()
            mapping=analyze(target_path,reference_path)
            base=working(apply_mapping(target,mapping,1))
            row['current_seconds']=time.perf_counter()-start
            candidate,timing=canonical.predict(target,reference); candidate=working(candidate)
            target_work,reference_work=working(target),working(reference)
            features,_,grid,ft=model.extract(target_work)
            ref_features,ref_colors,_,fr=model.extract(reference_work)
            with torch.inference_mode():
                anchors,confidence=match_anchors(features,ref_features,ref_colors)
                anchors=anchors.T.reshape(1,3,*grid); confidence=confidence.reshape(1,1,*grid)
                a,da=color_from_anchors(tensor(base,model.device),anchors,confidence)
                b,db=color_from_anchors(tensor(base,model.device),anchors,confidence,spatial=True)
                c,dc=choose_blend(tensor(base,model.device),tensor(candidate,model.device),anchors,confidence)
            outputs=dict(input=target_work,current=base,canonical=candidate,anchor_global=rgb8(a),anchor_spatial=rgb8(b),anchor_blend=rgb8(c))
            for name,image in outputs.items():
                path=prediction_dir/f'{index}-{name}.png'; Image.fromarray(image).save(path)
                row['outputs'][name]=dict(path=str(path.relative_to(out)),sha256=digest(path))
            row.update(original_shape=list(target.shape),working_shape=list(target_work.shape),
                source_sha256=dict(target=digest(target_path),reference=digest(reference_path)),
                global_diagnostics=da,spatial_diagnostics=db,blend=dc,canonical_time=timing,feature_time=dict(target=ft,reference=fr))
        except Exception as exc:
            row['failure']=dict(type=type(exc).__name__,message=str(exc))
        save(); print(json.dumps(dict(prediction_index=index,failure=row.get('failure'),confidence=row.get('global_diagnostics',{}).get('mean_confidence'))),flush=True)
    if len(cases)!=10 or tuple(x['index'] for x in cases)!=INDICES: raise ValueError('Incomplete prediction cohort')
    (out/'predictions-complete.json').write_text(json.dumps(dict(predictions_sha256=digest(out/'predictions.json'),
        count=len(cases),failed_indices=[x['index'] for x in cases if 'failure' in x],expert_targets_not_read=True),indent=2),encoding='utf-8')
    # Desired results are inaccessible locally until every prediction is frozen.
    with ThreadPoolExecutor(max_workers=3) as pool:
        truth_prepared=list(pool.map(prepare,[x for x in entries if x['path'].startswith('paired_gt/')]))
    (out/'expert-manifest.json').write_text(json.dumps(truth_prepared,indent=2),encoding='utf-8')
    scores=[]
    for row in cases:
        if 'failure' in row:
            scores.append(dict(index=row['index'],failure=row['failure'])); continue
        index=row['index']; truth_native=load(out/f'inputs/paired_gt/gt{index}.png')
        if list(truth_native.shape)!=row['original_shape']: raise ValueError('Expert target shape mismatch')
        truth=working(truth_native)
        images={}
        for name,record in row['outputs'].items():
            path=out/record['path']
            if digest(path)!=record['sha256']: raise ValueError('Prediction modified before scoring')
            images[name]=load(path)
        values={name:float(np.abs(image.astype(np.float32)-truth.astype(np.float32)).mean()) for name,image in images.items()}
        scores.append(dict(index=index,rgb_mae=values))
        panel=Image.new('RGB',(1080,780),'#222222'); draw=ImageDraw.Draw(panel)
        comparison=[('Input',images['input']),('Reference',working(load(out/f'inputs/paired_style/tar{index}.png'))),('Expert target',truth),
            ('Current',images['current']),('Canon',images['canonical']),('Anchor chosen blend',images['anchor_blend']),
            ('Anchor global',images['anchor_global']),('Anchor spatial',images['anchor_spatial'])]
        for k,(label,image) in enumerate(comparison):
            im=Image.fromarray(image); im.thumbnail((356,232)); x,y=k%3*360,k//3*260
            panel.paste(im,(x+(360-im.width)//2,y+24+(232-im.height)//2)); draw.text((x+8,y+6),label,fill='white')
        panel.save(out/f'{index}-comparison.jpg',quality=94)
    valid=[row for row in scores if 'rgb_mae' in row]; summary={}
    for name in ('input','current','canonical','anchor_global','anchor_spatial','anchor_blend'):
        before=np.array([row['rgb_mae']['current'] for row in valid]); actual=np.array([row['rgb_mae'][name] for row in valid])
        summary[name]=dict(count=len(valid),failed_count=10-len(valid),mean=float(actual.mean()) if len(valid) else None,
            median=float(np.median(actual)) if len(valid) else None,improved=int((actual<before-.1).sum()),
            regressed=int((actual>before+.1).sum()),severe=int(((actual>before+5)&(actual>before*1.25)).sum()))
    result=dict(scope=cohort['scope'],cohort_sha256=digest(out/'cohort.json'),predictions_sha256=digest(out/'predictions.json'),
        scores=scores,summary=summary,all_predictions_before_expert_download=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,output=str(out),summary=summary)),flush=True)


if __name__=='__main__': main()
