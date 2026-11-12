"""Development-only regional learned mapping test; oracle is used after prediction."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
from canonical_model import ROOT,CanonicalModel
from semantic_crops import SemanticCropCandidate
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze,algorithm_fingerprint
from reference_color_app.lut import apply_mapping


def score(output,truth,mask):
    error=np.abs(output.astype(float)-truth)
    fg=mask>.95;bg=mask<.05;edge=~(fg|bg)
    relation=None
    if fg.any() and bg.any():
        predicted=output[fg].mean(axis=0)-output[bg].mean(axis=0)
        expected=truth[fg].mean(axis=0)-truth[bg].mean(axis=0)
        relation=float(np.abs(predicted-expected).mean())
    return dict(mae=float(error.mean()),foreground_mae=float(error[fg].mean()) if fg.any() else None,
                background_mae=float(error[bg].mean()) if bg.any() else None,
                edge_mae=float(error[edge].mean()) if edge.any() else None,
                relation_error=relation,clipped_fraction=float(np.any((output==0)|(output==255),axis=2).mean()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=('crop','shared'),default='crop')
    args=parser.parse_args()
    data=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    jobfile=json.loads((data/'development/jobs.json').read_text(encoding='utf-8'))
    assert jobfile['split']=='development','Do not consume frozen evaluation data in development script'
    jobs=jobfile['jobs'];out=data/('development/semantic-candidate' if args.variant=='crop' else 'development/semantic-shared-candidate');out.mkdir(parents=True,exist_ok=True)
    model=CanonicalModel()
    if args.variant=='shared':
        from shared_semantic_crops import SharedCanonicalCandidate
        candidate=SharedCanonicalCandidate(model)
    else:candidate=SemanticCropCandidate(model)
    rows=[]
    for job in jobs:
        identifier,refid,profile=job['target'],job['reference'],job['profile']
        targetpath=data/f'{identifier}-input.png';referencepath=data/job['reference_file']
        target=np.array(Image.open(targetpath).convert('RGB'));reference=np.array(Image.open(referencepath).convert('RGB'))
        probabilities=data/'development/automatic-masks'
        tp=np.load(probabilities/f'{identifier}-target.npy',allow_pickle=False)
        rp=np.load(probabilities/f'{refid}-{profile}.npy',allow_pickle=False)
        result,global_rgb,diagnostics=candidate.predict(target,reference,tp,rp)
        baseline=apply_mapping(target,analyze(targetpath,referencepath),1.)
        # Human annotations and desired target output are not accessible to the
        # candidate; load them only after all three predictions finish.
        truth=np.array(Image.open(data/job['truth_file']).convert('RGB'))
        mask=np.asarray(Image.open(data/f'{identifier}-human-mask.png').filter(ImageFilter.GaussianBlur(2)),dtype=float)/255
        row=dict(target=identifier,reference=refid,group=job['group'],profile=profile,
                 baseline=score(baseline,truth,mask),model=score(global_rgb,truth,mask),candidate=score(result,truth,mask),diagnostics=diagnostics)
        rows.append(row)
        name=f'{identifier}-{refid}-{profile}'
        for label,rgb in [('global',global_rgb),('candidate',result)]:Image.fromarray(rgb).save(out/f'{name}-{label}.png')
        panel=Image.new('RGB',(5*256,288),'#202020');draw=ImageDraw.Draw(panel)
        for k,(label,rgb) in enumerate(zip(('target','reference','known control','global model','regional candidate'),[target,reference,truth,global_rgb,result])):
            image=Image.fromarray(rgb);image.thumbnail((256,256),Image.Resampling.BILINEAR)
            panel.paste(image,(k*256,32));draw.text((k*256+4,8),label,fill='white')
        panel.save(out/f'{name}-comparison.jpg')
    def summarize(group):
        relational=[r for r in group if r['candidate']['relation_error'] is not None]
        return dict(count=len(group),applied=sum(r['diagnostics']['applied'] for r in group),
                    improved_over_model=sum(r['candidate']['mae']<r['model']['mae']-.1 for r in group),
                    worsened_over_model=sum(r['candidate']['mae']>r['model']['mae']+.1 for r in group),
                    median_baseline=float(np.median([r['baseline']['mae'] for r in group])),
                    median_model=float(np.median([r['model']['mae'] for r in group])),
                    median_candidate=float(np.median([r['candidate']['mae'] for r in group])),
                    median_model_relation=float(np.median([r['model']['relation_error'] for r in relational])) if relational else None,
                    median_candidate_relation=float(np.median([r['candidate']['relation_error'] for r in relational])) if relational else None)
    report=dict(scope='Six project-new COCO development pairs; synthetic known styles, not natural aesthetics or frozen evaluation',
                algorithm_fingerprint=algorithm_fingerprint(),model=model.provenance,
                variant=args.variant,
                candidate_source_sha256=hashlib.sha256(Path(__file__).with_name('semantic_crops.py' if args.variant=='crop' else 'shared_semantic_crops.py').read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256(Path(__file__).with_name('semantic-crop-protocol.md' if args.variant=='crop' else 'shared-canonical-protocol.md').read_bytes()).hexdigest(),
                summary=summarize(rows),by_profile={p:summarize([r for r in rows if r['profile']==p]) for p in sorted({r['profile'] for r in rows})},
                by_group={g:summarize([r for r in rows if r['group']==g]) for g in ('single','multiple','none')},cases=rows)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','model')},indent=2))


if __name__=='__main__':main()
