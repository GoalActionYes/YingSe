"""Author-model baseline on exactly the frozen cross-scene inputs; no training."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from canonical_model import CanonicalModel

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    directory=ROOT/'outputs/adaptive-reference-20261113/cross-scene'
    prior=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    cases=prior['cases'][:args.limit] if args.limit else prior['cases']
    out=ROOT/'outputs/adaptive-reference-20261113/canonical-model'
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); model=CanonicalModel(); load_seconds=time.perf_counter()-start
    rows=[]
    with (out/'progress.jsonl').open('w',encoding='utf-8') as stream:
        for case in cases:
            i,j,name=case['target'],case['reference'],case['style']
            target=np.asarray(Image.open(directory/f'target-{i}.png').convert('RGB'))
            reference=np.asarray(Image.open(directory/f'reference-{j}-{name}.png').convert('RGB'))
            # Same transform of target was saved among reference assets; it is
            # accessed only AFTER inference and never passed to the model.
            result,timing=model.predict(target,reference)
            desired=np.asarray(Image.open(directory/f'reference-{i}-{name}.png').convert('RGB'))
            error=float(np.abs(result.astype(float)-desired).mean())
            Image.fromarray(result).save(out/f'{i}-{j}-{name}.png')
            row=dict(target=i,reference=j,style=name,model_mae=error,
                     global_mae=case['baseline_mae'],regional_mae=case['candidate_mae'],**timing)
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
    def summarize(group):
        return dict(count=len(group),improved_over_global=sum(r['model_mae']<r['global_mae']-.1 for r in group),
                    worsened_over_global=sum(r['model_mae']>r['global_mae']+.1 for r in group),
                    median_model_mae=float(np.median([r['model_mae'] for r in group])),
                    median_global_mae=float(np.median([r['global_mae'] for r in group])))
    report=dict(scope='Author E2E model, same historical synthetic-style counter-test; not independent natural quality',
                partial_probe=bool(args.limit),load_seconds=load_seconds,provenance=model.provenance,
                summary=summarize(rows),by_style={name:summarize([r for r in rows if r['style']==name])
                                                for name in sorted({r['style'] for r in rows})},cases=rows)
    (out/('smoke-results.json' if args.limit else 'results.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
