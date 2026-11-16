"""Read-only triplet-schema and exact-path conflict audit; no image consumption."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from independent_coco import ROOT,digest


def inspect(rows):
    labels=defaultdict(set);counts=Counter()
    for row in rows:
        if not all(isinstance(row.get(k),str) and row[k] for k in ['content','reference','gt']):raise ValueError('Invalid triplet')
        labels[(row['content'],row['reference'])].add(row['gt']);counts[tuple(sorted(row.keys()))]+=1
    return dict(rows=len(rows),unique_input_path_pairs=len(labels),exact_input_pairs_with_multiple_gt_paths=sum(len(x)>1 for x in labels.values()),
        schemas=[dict(keys=list(k),rows=n) for k,n in sorted(counts.items())])


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';p=ROOT/'.tools/TST100K-index/triplet.json';cohort=json.loads((base/'cohort-multiref-v1/cohort.json').read_text())
    assert digest(p)==cohort['index_sha256'];rows=json.loads(p.read_text())
    report=dict(scope='Metadata-only supervision task audit; paths are not pixels or semantic truth',index_sha256=digest(p),source_sha256=digest(Path(__file__)),all_index=inspect(rows),cohort={})
    all_targets=defaultdict(set)
    for row in rows:all_targets[(row['content'],row['reference'])].add(row['gt'])
    for split in ['train','validation']:
        part=cohort['groups'][split]
        for row in part:assert all(row[k]==rows[row['index']][k] for k in ['content','reference','gt'])
        report['cohort'][split]=inspect([{k:r[k] for k in ['content','reference','gt']} for r in part])
        report['cohort'][split]['input_pairs_with_alternative_gt_in_full_index']=[dict(index=r['index'],gt_paths=sorted(all_targets[(r['content'],r['reference'])])) for r in part if len(all_targets[(r['content'],r['reference'])])>1]
    report['limitations']=['Different GT paths may have identical pixels; conflicts are only candidates for investigation','No duplicate-path conflict does not prove perfect perceptual labels','No masks, material identity or should-transfer labels present in three-path schema','No independent evaluation images read']
    (ROOT/'docs/research/2026-11-17-supervision-metadata-evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
