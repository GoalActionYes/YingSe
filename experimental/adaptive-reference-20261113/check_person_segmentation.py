"""Development-only human-mask comparison; does not feed oracle masks to inference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from canonical_model import ROOT
from person_segmenter import PersonSegmenter


def main():
    data=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    jobs=json.loads((data/'development/jobs.json').read_text(encoding='utf-8'))['jobs']
    out=data/'development/automatic-masks';out.mkdir(parents=True,exist_ok=True)
    engine=PersonSegmenter();rows=[]
    inputs={}
    for job in jobs:
        inputs[f"{job['target']}-target"]=(job['target'],data/f"{job['target']}-input.png",job['group'])
        inputs[f"{job['reference']}-{job['profile']}"]=(job['reference'],data/job['reference_file'],job['group'])
    for name,(identifier,path,group) in inputs.items():
        rgb=np.array(Image.open(path).convert('RGB'))
        prediction,timing=engine.predict(rgb)
        # Access independent human annotation only after automatic prediction.
        oracle=np.asarray(Image.open(data/f'{identifier}-human-mask.png'))>0
        predicted=prediction>.5;union=np.count_nonzero(predicted|oracle)
        iou=float(np.count_nonzero(predicted&oracle)/union) if union else 1.
        np.save(out/(name+'.npy'),prediction.astype(np.float32),allow_pickle=False)
        rows.append(dict(name=name,group=group,iou=iou,false_positive_fraction=float((predicted&~oracle).mean()),
                         human_fraction=float(oracle.mean()),**timing))
    report=dict(scope='36 development inputs including styled references; not held-out evaluation',
                provenance=engine.provenance,cases=rows,
                by_group={group:dict(count=sum(r['group']==group for r in rows),
                                     median_iou=float(np.median([r['iou'] for r in rows if r['group']==group])),
                                     max_false_positive_fraction=max(r['false_positive_fraction'] for r in rows if r['group']==group))
                          for group in ('single','multiple','none')})
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
