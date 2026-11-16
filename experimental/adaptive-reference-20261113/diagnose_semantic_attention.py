"""RGB-only frozen attention statistics for the complete consumed cohort."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from semantic_local import match


def main():
    torch.set_num_threads(4)
    b=ROOT/'outputs/adaptive-reference-20261113/tst100k';cache=b/'semantic-token-inputs-v1';p=cache/'results.json';r=json.loads(p.read_text());assert r['finished']
    registry_path=b/'regional-correction-inputs-v1/results.json';assert digest(registry_path)==r['registry_sha256'];registry=json.loads(registry_path.read_text());inputs={e['index']:e for e in r['inputs']}
    train_path=b/'semantic-local-training-v1/results.json';training=json.loads(train_path.read_text());assert training['finished'] and training['token_report_sha256']==digest(p)
    out=b/'semantic-attention-diagnostic-v1';out.mkdir(exist_ok=False)
    report=dict(scope='Frozen RGB-only attention statistics, not semantic accuracy or quality',cache_results_sha256=digest(p),training_results_sha256=digest(train_path),source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['diagnose_semantic_attention.py','semantic-attention-diagnostic-protocol.md','semantic_local.py']},cases=[],finished=False)
    tokens={}
    for e in r['inputs']:
        path=cache/e['path'];assert digest(path)==e['sha256']
        with np.load(path,allow_pickle=False) as a:tokens[e['index']]=torch.from_numpy(np.concatenate([a['features'],a['colors']],axis=1))[None]
    maximum=0.
    with torch.inference_mode():
        for row in registry['rows']:
            c,z=[tokens[row['roles'][role]['feature_index']] for role in ['content','reference']]
            scores=torch.bmm(F.normalize(c[...,:384],dim=-1),F.normalize(z[...,:384],dim=-1).transpose(1,2))*10
            w=(scores-scores.max(-1,keepdim=True).values).exp();w=w/w.sum(-1,keepdim=True)
            actual=torch.bmm(w,z);difference=float((actual-match(c,z)).abs().max());maximum=max(maximum,difference);assert difference<=1e-6
            entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)
            uniform=z.mean(1,keepdim=True).expand_as(actual)
            report['cases'].append(dict(index=row['index'],split=row['split'],mean_max_weight=float(w.max(-1).values.mean()),normalized_entropy=float((entropy/np.log(64)).mean()),effective_tokens=float(entropy.exp().mean()),matched_vs_uniform_rgb_mae=float((actual[...,384:]-uniform[...,384:]).abs().mean())*255,
                matched_color_spatial_std=float(actual[...,384:].std(dim=1,correction=0).mean())*255,reference_color_spatial_std=float(z[...,384:].std(dim=1,correction=0).mean())*255))
    report['summary']={}
    for split in ['train','validation']:
        rows=[x for x in report['cases'] if x['split']==split];report['summary'][split]={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ['index','split']}
    report['training_log']={arm:dict(losses=[x['loss'] for x in a['epochs']],development_rgb=[x['mean'] for x in a['epochs']],best_epoch=a['best_epoch']) for arm,a in training['arms'].items()}
    report['maximum_match_recalculation_error']=maximum;report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2));(ROOT/'docs/research/2026-11-17-semantic-attention-evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(summary=report['summary'],maximum_match_recalculation_error=maximum,training_log=report['training_log'])),flush=True)


if __name__=='__main__':main()
