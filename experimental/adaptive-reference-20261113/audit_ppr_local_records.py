"""Locate explicit local-edit structures among existing provisional candidates only."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from independent_coco import ROOT,digest

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-spatial-information-v1/local-record-availability.json'


def main():
    source=BASE/'ppr-group-exclusions-v3/results.json'
    assert digest(source)=='8783deef13634bd1433964d251e77af3cba765c280c5b4741cdf13318263f716'
    links=json.loads(source.read_text());rows=[r for r in links['training_candidates'] if not r['exclude_for_known_consumed_group']]
    assert len(rows)==479
    path=ROOT/'.tools/PPR10K-audit/xmp_target_a.zip'
    assert digest(path)=='b02e46d25ef0aeee4c6c6a1bc5dc517066bf8b009696a70d7db1f48083d04a7a'
    ns='{http://ns.adobe.com/camera-raw-settings/1.0/}'
    fields={'PaintBasedCorrections','CorrectionMasks','GradientBasedCorrections','CircularGradientBasedCorrections'}
    found=[]
    with zipfile.ZipFile(path) as z:
        lookup={Path(n).stem:n for n in z.namelist() if n.endswith('.xmp')}
        for row in rows:
            xml=ET.fromstring(z.read(lookup[row['ppr_candidate_id']]))
            present=set()
            for node in xml.iter():
                if node.tag.startswith(ns):present.add(node.tag[len(ns):])
                present.update(k[len(ns):] for k in node.attrib if k.startswith(ns))
            hit=present&fields
            if hit:found.append(dict(ppr_candidate_id=row['ppr_candidate_id'],ppr_group=row['ppr_group'],source_path=row['path'],local_fields=sorted(hit)))
    report=dict(scope='cached metadata availability only; not pixel or effect qualification',source_sha256=digest(source),driver_sha256=digest(__file__),eligible_provisional_scanned=479,explicit_local_record_count=len(found),rows=found,training_eligibility_established=False)
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
