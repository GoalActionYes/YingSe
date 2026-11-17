"""Recompute all completed screen scores/inputs and verify bridge/stopping."""
import argparse
import hashlib
import json
from pathlib import Path

from screen_ictone_first import ROOT,digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('.','..')
    import numpy as np
    from PIL import Image,ImageDraw
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    out=base/args.run
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=load(out/'results.json')
    assert report['finished']
    for name,sha in report['source_sha256'].items():
        frozen=out/name
        if not frozen.exists():
            assert digest(Path(__file__).with_name(name))==sha
            frozen.write_bytes(Path(__file__).with_name(name).read_bytes())
        assert digest(frozen)==sha
    protocol=out/'protocol-frozen.md'
    if not protocol.exists():
        path=ROOT/'docs/research/2026-11-17-ictone-screen.md'
        assert digest(path)==report['protocol_sha256']
        protocol.write_bytes(path.read_bytes())
    assert digest(protocol)==report['protocol_sha256']
    bridge=base/report['bridge']['run']
    assert digest(bridge/'results.json')==report['bridge']['producer_sha256']
    assert digest(bridge/'audit.json')==report['bridge']['audit_sha256']
    bridge_report=load(bridge/'results.json')
    assert load(bridge/'audit.json')['state_parameters_verified']==1846
    for name in ['triptych-output.png','native512.png','metric256.png']:
        with Image.open(base/'ictone-first-v1'/name) as im:a=np.array(im)
        with Image.open(bridge/name) as im:b=np.array(im)
        assert np.array_equal(a,b)
    data=base/'cohort-training-coverage-v2'
    assert digest(data/'cohort.json')=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    assert digest(data/'srgb-v1/manifest.json')=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    rows=load(data/'cohort.json')['groups']['validation'][:8]
    assert report['order']==[x['index'] for x in rows]
    lookup={x['original_path']:x for x in load(data/'srgb-v1/manifest.json')['files']}
    baseline={x['index']:x for x in load(base/'training-coverage-expanded-v2/results.json')['baseline']['cases']}
    def read(row,role):
        item=lookup[row[role]]
        path=data/item['prepared_path']
        assert path.resolve().is_relative_to(base.resolve()) and digest(path)==item['sha256']
        with Image.open(path) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    completed=report['completed']
    assert [x['index'] for x in completed]==report['order'][:len(completed)]
    severe=0
    recomputed=[]
    panel=Image.new('RGB',(1280,282*len(completed)),'white')
    draw=ImageDraw.Draw(panel)
    for ordinal,item in enumerate(completed):
        row=rows[ordinal]
        folder=base/item['run']
        child=load(folder/'results.json')
        assert digest(folder/'results.json')==item['producer_sha256']
        assert child['finished'] and child['index']==item['index'] and child['completed_steps']==4
        assert child['config']==report['config'] and child['source_sha256']==bridge_report['source_sha256']
        assert child['checked_parameter_count']==1846 and child['reused_stage_from']=='ictone-first-v1'
        for name,sha in child['source_sha256'].items():assert digest(folder/name)==sha
        for name,sha in child['output_sha256'].items():assert digest(folder/name)==sha
        for role in ['content','reference']:
            with Image.open(folder/(role+'256.png')) as im:
                assert np.array_equal(np.array(im),np.array(read(row,role)))
            assert digest(folder/(role+'256.png'))==child['inputs'][role]['sha256']
        with Image.open(folder/'triptych-output.png') as im:
            assert im.size==(1536,512)
            native=im.crop((1024,0,1536,512))
        with Image.open(folder/'native512.png') as im:assert np.array_equal(np.array(im),np.array(native))
        metric=native.resize((256,256),Image.Resampling.BILINEAR)
        with Image.open(folder/'metric256.png') as im:assert np.array_equal(np.array(im),np.array(metric))
        path=base/'multiref-control-v1/epoch-6'/f"{row['index']}.png"
        assert digest(path)==baseline[row['index']]['sha256']
        with Image.open(path) as im:old=im.copy()
        truth=read(row,'gt')
        t=np.array(truth).astype(float)
        old_mae=float(np.abs(np.array(old).astype(float)-t).mean())
        mae=float(np.abs(np.array(metric).astype(float)-t).mean())
        is_severe=bool(mae-old_mae>5 and mae>old_mae*1.25)
        assert old_mae==item['baseline_mae']==baseline[row['index']]['mae'] and mae==item['candidate_mae']
        assert is_severe==item['severe'] and digest(folder/'metric256.png')==item['metric_sha256']
        severe+=is_severe
        if ordinal<len(completed)-1:assert severe<3,'generated cases after mandatory stop'
        recomputed.append(dict(index=row['index'],baseline_mae=old_mae,candidate_mae=mae,severe=is_severe))
        for column,(label,picture) in enumerate([('Content',read(row,'content')),('Reference',read(row,'reference')),
                ('GT offline',truth),('current13',old),('ICTone Q3 README',metric)]):
            top=ordinal*282
            panel.paste(picture,(256*column,top+26))
            draw.text((256*column+5,top+7),str(row['index'])+' '+label,fill='black')
    assert severe==report['severe_count']
    assert report['all_eight_completed']==(len(completed)==8)
    if report['stop_reason']=='necessary_severe_gate':
        assert severe==3 and completed[-1]['severe'] and report['full64_severe_gate_impossible']
    else:
        assert report['stop_reason']=='all_eight_completed' and len(completed)==8 and severe<3
        assert not report['full64_severe_gate_impossible']
    panel.save(out/'diagnostic-panel.png')
    result=dict(finished=True,scope='completed consumed subset only; no full64/independent/human acceptance',
                producer_sha256=digest(out/'results.json'),source_sha256=digest(Path(__file__)),
                scores=recomputed,all_inputs_and_crops_exact=True,bridge_pixels_exact=True,
                child_state_identity_checks='1846 per child producer observed; first bridge independently file-audited',
                completed_count=len(completed),severe_count=severe,stop_reason=report['stop_reason'],
                candidate_mean=sum(x['candidate_mae'] for x in completed)/len(completed),
                baseline_mean=sum(x['baseline_mae'] for x in completed)/len(completed),
                improvement_count=sum(x['baseline_mae']-x['candidate_mae']>.1 for x in completed))
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
