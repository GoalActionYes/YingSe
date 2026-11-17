"""Inspect safe structural predicates only; never claim an Adobe rendering result."""
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile
from independent_coco import ROOT,digest

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-local-effect-structure-v2'
NS='{http://ns.adobe.com/camera-raw-settings/1.0/}'
RDF='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
COLOR=('Exposure','Brightness','Saturation','Temperature','Tint','Contrast','Highlights','Shadows','Whites','Blacks','Toning')
DETAIL=('Sharpness','Clarity','Texture','Dehaze','Noise','Moire','Defringe')


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    availability=BASE/'ppr-spatial-information-v1/local-record-availability.json'
    data=json.loads(availability.read_text());assert data['eligible_provisional_scanned']==479 and data['explicit_local_record_count']==3
    ids=sorted((r['ppr_candidate_id'] for r in data['rows']),key=lambda s:tuple(map(int,s.split('_'))))
    assert ids==['244_4','1369_1','1645_1']
    path=ROOT/'.tools/PPR10K-audit/xmp_target_a.zip'
    assert digest(path)=='b02e46d25ef0aeee4c6c6a1bc5dc517066bf8b009696a70d7db1f48083d04a7a'
    protocol=Path(__file__).with_name('ppr-local-effect-protocol.md')
    config=dict(ids=ids,availability_sha256=digest(availability),archive_sha256=digest(path),driver_sha256=digest(__file__),protocol_sha256=digest(protocol))
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (OUT/'protocol.md').write_bytes(protocol.read_bytes())
    rows=[]
    with zipfile.ZipFile(path) as z:
        for key in ids:
            assert time.monotonic()-start<60
            members=[n for n in z.namelist() if Path(n).stem==key];assert len(members)==1
            xml=ET.fromstring(z.read(members[0]));parents=[n for n in xml.iter() if n.tag==NS+'PaintBasedCorrections'];corrections=[]
            for parent in parents:
                for node in parent.iter():
                    if NS+'CorrectionActive' not in node.attrib:continue
                    attrs={n[len(NS):]:v for n,v in node.attrib.items() if n.startswith(NS)}
                    active={'true':True,'1':True,'false':False,'0':False}.get(attrs['CorrectionActive'].strip())
                    amount_positive=float(attrs.get('CorrectionAmount','0'))>0
                    nonzero=[]
                    for name,value in attrs.items():
                        if not name.startswith('Local'):continue
                        number=float(value);assert abs(number)<1e8
                        if number!=0:nonzero.append(name)
                    color=[n for n in nonzero if any(s in n for s in COLOR)]
                    detail=[n for n in nonzero if any(s in n for s in DETAIL) and n not in color]
                    unknown=sorted(set(nonzero)-set(color)-set(detail))
                    brushes=[]
                    for mask in node.iter():
                        if NS+'MaskValue' not in mask.attrib:continue
                        mask_attrs={n[len(NS):]:v for n,v in mask.attrib.items() if n.startswith(NS)}
                        dabs=sum(1 for d in mask.iter() if d.tag==RDF+'li' and d.text and d.text.strip())
                        positive=all(float(mask_attrs.get(n,'0'))>0 for n in ['MaskValue','Flow','Radius'])
                        brushes.append(dict(positive_mask_flow_radius=positive,nonempty_dabs=dabs))
                    eligible=active is True and amount_positive and bool(color) and any(b['positive_mask_flow_radius'] and b['nonempty_dabs']>0 for b in brushes)
                    corrections.append(dict(active_value=active,amount_positive=amount_positive,nonzero_color_fields=sorted(color),nonzero_detail_fields=sorted(detail),nonzero_unknown_fields=unknown,brushes=brushes,metadata_retrieval_predicate=bool(eligible)))
            rows.append(dict(id=key,corrections=corrections,metadata_retrieval_predicate=any(c['metadata_retrieval_predicate'] for c in corrections)))
    report=dict(finished=True,config_sha256=digest(OUT/'config.json'),rows=rows,eligible_for_pixel_check=[r['id'] for r in rows if r['metadata_retrieval_predicate']],actual_rendered_local_effect_proven=False,product_candidate_trained=False,private_identifiers_or_coordinates_exported=False,seconds=time.monotonic()-start)
    assert digest(__file__)==config['driver_sha256']
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
