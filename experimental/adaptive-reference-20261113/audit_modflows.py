"""Recompute pinned default-screen scores and gates from all saved PNGs."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'modflows-screen-v1';load=lambda p:json.loads(p.read_text());r=load(out/'results.json');assert r['finished'] and r['model_state_unchanged'] and r['first_exact_repeat']
    assets=ROOT/'.tools/ModFlows-audit'
    for e in r['author_source']['files']:assert digest(assets/'source'/e['path'])==e['sha256']
    assert digest(assets/r['weight']['name'])==r['weight']['sha256']
    for e in r['dependencies']:assert digest(assets/'wheels'/e['filename'])==e['sha256']
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=base/'cohort-training-coverage-v2';assert digest(data/'cohort.json')==r['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==r['manifest_sha256']
    rows=load(data/'cohort.json')['groups']['validation'];lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']};truths={};pred={};baseline={};input_arrays={};byindex={e['index']:e for e in r['rows']}
    old=base/'multiref-control-v1/epoch-6';original=load(base/'training-coverage-expanded-v2/results.json')['baseline'];hashes={e['index']:e['sha256'] for e in original['cases']}
    for row in rows:
        index=row['index'];e=lookup[row['gt']];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:truths[index]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        path=out/'candidate'/f'{index}.png';assert digest(path)==byindex[index]['sha256']
        with Image.open(path) as im:pred[index]=np.array(im)
        path=old/f'{index}.png';assert digest(path)==hashes[index]
        with Image.open(path) as im:baseline[index]=np.array(im)
        input_arrays[index]={}
        for role,e in byindex[index]['inputs'].items():
            p=out/e['path'];assert digest(p)==e['sha256']
            with Image.open(p) as im:a=np.array(im)
            assert hashlib.sha256(a.tobytes()).hexdigest()==e['pixel_sha256'];input_arrays[index][role]=a
    actual=score_outputs(rows,pred,truths);start=score_outputs(rows,baseline,truths)
    for e in actual['cases']:e['sha256']=digest(out/'candidate'/f"{e['index']}.png")
    assert actual==r['scores'] and start==r['baseline'] and response_development_gate(start,actual,start)==r['gate']
    oldmae={e['index']:e['mae'] for e in start['cases']};delta={e['index']:oldmae[e['index']]-e['mae'] for e in actual['cases']}
    chosen=list(dict.fromkeys([rows[0]['index'],rows[1]['index'],max(delta,key=delta.get),min(delta,key=delta.get)]))
    panel=Image.new('RGB',(5*256,len(chosen)*280),(245,245,245));draw=ImageDraw.Draw(panel)
    for j,index in enumerate(chosen):
        arrays=[input_arrays[index]['content'],input_arrays[index]['reference'],truths[index],baseline[index],pred[index]]
        for i,(name,a) in enumerate(zip(['Content','Reference','GT','Current13','ModFlows'],arrays)):
            draw.text((i*256+4,j*280+4),str(index)+' '+name,fill='black');panel.paste(Image.fromarray(a),(i*256,j*280+24))
    path=out/'diagnostic-panel.png';panel.save(path)
    evidence=dict(results_sha256=digest(out/'results.json'),auditor_sha256=digest(Path(__file__)),candidate_pngs_recomputed=64,baseline_pngs_recomputed=64,input_pngs_verified=128,all_scores_and_gate_equal=True,diagnostic_case_indices=chosen,panel_sha256=digest(path),mean=actual['mean'],baseline_mean=start['mean'],gate=r['gate'],training=False,product_integrated=False)
    (ROOT/'docs/research/2026-11-17-modflows-audit-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print(json.dumps(evidence))


if __name__=='__main__':main()
