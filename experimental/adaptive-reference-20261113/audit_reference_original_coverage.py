"""Check locally available true reference-original training pairs; no GT reads."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    index=ROOT/'.tools/TST100K-index/triplet.json'
    assert digest(index)=='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413'
    originals=defaultdict(set)
    for row in json.loads(index.read_text()):originals[Path(row['content']).stem].add(row['content'])
    evidence=dict(scope='Training-only reference-before/after availability and pixels; no target GT, fitting, training or quality evaluation',cohorts={},source_sha256=digest(Path(__file__)))
    private=base/'reference-original-coverage-v1';private.mkdir(exist_ok=False)
    for name,expected in [('cohort-multiref-v1',(75,47,73)),('cohort-training-coverage-v2',(865,438,829))]:
        data=base/name;cohort_path=data/'cohort.json';manifest_path=data/'srgb-v1/manifest.json'
        cohort=json.loads(cohort_path.read_text());manifest=json.loads(manifest_path.read_text())
        assert manifest['passed'] and manifest['cohort_sha256']==digest(cohort_path)
        if name.endswith('v2'):assert digest(cohort_path)=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
        lookup={e['original_path']:e for e in manifest['files']};train={r['content'] for r in cohort['groups']['train']}
        selected=[]
        for r in cohort['groups']['train']:
            opts=originals[r['reference_family']]
            if len(opts)==1:
                original=next(iter(opts))
                if original in train and original in lookup:selected.append((r['index'],original,r['reference']))
        pairs=sorted({(a,b) for _,a,b in selected})
        assert (len(selected),len({a for _,a,_ in selected}),len(pairs))==expected
        cache={};records=[]
        for a,b in pairs:
            arrays=[]
            for p in [a,b]:
                if p not in cache:
                    e=lookup[p];path=data/e['prepared_path'];assert digest(path)==e['sha256']
                    with Image.open(path) as im:
                        size=list(im.size);rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                    cache[p]=(rgb,size,e['sha256'])
                arrays.append(cache[p][0])
            records.append(dict(original=a,reference=b,original_sha256=cache[a][2],reference_sha256=cache[b][2],
                original_size=cache[a][1],reference_size=cache[b][1],same_size=cache[a][1]==cache[b][1],
                rgb256_mae=float(np.abs(arrays[0].astype(float)-arrays[1]).mean()),pixel_equal=bool(np.array_equal(*arrays))))
        path=private/(name+'.json');path.write_text(json.dumps(dict(selected=selected,pairs=records),indent=2))
        scores=np.array([r['rgb256_mae'] for r in records])
        evidence['cohorts'][name]=dict(cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(manifest_path),eligible_rows=len(selected),
            unique_originals=len({a for _,a,_ in selected}),unique_before_after_pairs=len(pairs),checked_files=len(cache),
            same_size_pairs=sum(r['same_size'] for r in records),pixel_equal_pairs=sum(r['pixel_equal'] for r in records),
            before_after_mae_mean=float(scores.mean()),before_after_mae_median=float(np.median(scores)),
            detailed_training_report_sha256=digest(path),all_originals_are_training_contents=True)
        print(json.dumps(dict(cohort=name,**evidence['cohorts'][name])),flush=True)
    evidence['finished']=True
    (private/'results.json').write_text(json.dumps(evidence,indent=2))
    (ROOT/'docs/research/2026-11-17-reference-original-coverage-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')


if __name__=='__main__':main()
