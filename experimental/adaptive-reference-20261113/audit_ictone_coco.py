"""Offline six-case integrity, RGB/region scores and stop-rule audit."""
import argparse
import json
import hashlib
from pathlib import Path

from ictone_flux_stage import ROOT
from screen_ictone_first import digest
from reconstruct_ictone_lut import author_functions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('.','..')
    import numpy as np
    from PIL import Image,ImageDraw
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    inputs=base/'ictone-coco-inputs-v1'
    out=base/args.run
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    report=load(out/'results.json')
    assert report['finished']
    plan=load(inputs/'plan.json')
    assert digest(inputs/'plan.json')==report['plan_sha256']
    assert digest(out/'protocol-frozen.md')==report['protocol_sha256']==plan['protocol_sha256']
    for name,sha in report['source_sha256'].items():assert digest(out/name)==sha
    manifest=load(coco/'manifest.json')
    provenance=load(ROOT/'outputs/adaptive-reference-20261113/ictone-cross-source-audit-v2/results.json')
    assert digest(coco/'manifest.json')==provenance['coco_manifest_sha256']
    image_metadata={x['id']:x for x in manifest['images']}
    funcs=author_functions()
    completed=report['completed']
    assert [x['index'] for x in completed]==[x['index'] for x in plan['cases'][:len(completed)]]
    panel=Image.new('RGB',(1536,282*len(completed)),'white')
    draw=ImageDraw.Draw(panel)
    measurements=[]
    application_error=0
    severe=0
    def read(path):
        with Image.open(path) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    def close_scores(a,b):
        assert set(a)==set(b)
        for k in a:
            if a[k] is None or b[k] is None:assert a[k] is None and b[k] is None
            else:assert abs(a[k]-b[k])<=1e-12,(k,a[k],b[k])
    for ordinal,(record,item) in enumerate(zip(completed,plan['cases'])):
        index=item['index']
        folder=base/record['child_run']
        child=load(folder/'results.json')
        assert digest(folder/'results.json')==record['producer_sha256']
        assert child['finished'] and child['index']==index and child['completed_steps']==4
        assert child['checked_parameter_count']==1846 and child['reused_stage_from']=='ictone-first-v1'
        assert child['config']==dict(steps=4,guidance=50.,seed=666,panel=[512,512],source='author pinned README')
        for name,sha in child['source_sha256'].items():assert digest(folder/name)==sha==report['source_sha256'][name]
        assert digest(Path(item['spec_path']))==item['spec_sha256']==child['input_spec_sha256']
        spec=load(Path(item['spec_path']))
        assert set(spec)=={'index','content','reference'}
        for role in ['content','reference']:
            assert digest(folder/(role+'256.png'))==spec[role]['sha256']
        for name,sha in child['output_sha256'].items():assert digest(folder/name)==sha
        content=read(folder/'content256.png')
        reference=read(folder/'reference256.png')
        if item['profile']=='same_image':assert np.array_equal(np.array(content),np.array(reference))
        original=read(folder/'native512.png')
        with Image.open(folder/'triptych-output.png') as im:
            assert im.size==(1536,512)
            crop=im.crop((1024,0,1536,512))
        with Image.open(folder/'native512.png') as im:assert np.array_equal(np.array(im),np.array(crop))
        with Image.open(folder/'metric256.png') as im:assert np.array_equal(np.array(im),np.array(original))
        projected=read(out/(index+'.png'))
        assert digest(out/(index+'.png'))==record['output_sha256']
        assert digest(out/(index+'-lut.npy'))==record['lut_sha256']
        lut=np.load(out/(index+'-lut.npy'),allow_pickle=False)
        assert lut.shape==(33,33,33,3) and np.isfinite(lut).all()
        portable=funcs['apply_lut'](np.array(content),lut,device='cpu')
        error=int(np.abs(portable.astype(int)-np.array(projected).astype(int)).max())
        assert error<=1,error
        application_error=max(application_error,error)
        assert digest(Path(item['truth_source']))==item['truth_sha256']
        assert digest(Path(item['baseline_path']))==item['baseline_sha256']
        assert digest(Path(item['human_mask_source']))==image_metadata[item['target']]['mask_sha256']
        truth=read(item['truth_source'])
        baseline=read(item['baseline_path'])
        with Image.open(item['human_mask_source']) as im:mask=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
        t=np.array(truth).astype(float)
        def score(im):
            rgb=np.array(im)
            err=np.abs(rgb.astype(float)-t).mean(2)
            return dict(mae=float(err.mean()),person_mae=float(err[mask].mean()) if mask.any() else None,
                        background_mae=float(err[~mask].mean()) if (~mask).any() else None,
                        clipped_channel_fraction=float(((rgb==0)|(rgb==255)).mean()))
        old,raw,new=score(baseline),score(original),score(projected)
        close_scores(old,record['baseline']);close_scores(raw,record['raw']);close_scores(new,record['candidate'])
        is_severe=bool(new['mae']-old['mae']>5 and new['mae']>old['mae']*1.25)
        assert is_severe==record['severe']
        severe+=is_severe
        if ordinal<len(completed)-1:assert severe<3,'generated after mandatory stop'
        measurements.append(dict(index=index,profile=item['profile'],baseline=old,raw=raw,candidate=new,severe=is_severe))
        for column,(label,picture) in enumerate([('Content',content),('Reference',reference),('GT offline',truth),
                ('current13',baseline),('Raw Q3',original),('LUT33',projected)]):
            y=282*ordinal
            panel.paste(picture,(column*256,y+26))
            draw.text((column*256+4,y+7),index+' '+label,fill='black')
    assert severe==report['severe_count']
    assert report['all_six_completed']==(len(completed)==6)
    if report['stop_reason']=='necessary_severe_gate':assert severe==3 and completed[-1]['severe']
    else:assert report['stop_reason']=='all_six_completed' and len(completed)==6 and severe<3
    panel.save(out/'diagnostic-panel.png')
    result=dict(finished=True,scope='consumed controlled cross-source subset; no natural/independent/product acceptance',
                source_sha256=digest(Path(__file__)),producer_sha256=digest(out/'results.json'),rows=measurements,
                count=len(completed),severe_count=severe,stop_reason=report['stop_reason'],
                baseline_mean=sum(x['baseline']['mae'] for x in measurements)/len(measurements),
                candidate_mean=sum(x['candidate']['mae'] for x in measurements)/len(measurements),
                improvement_count=sum(x['baseline']['mae']-x['candidate']['mae']>.1 for x in measurements),
                same_image_changes=[dict(index=x['index'],mae=x['candidate']['mae']) for x in measurements if x['profile']=='same_image'],
                input_truth_mask_score_checks=True,stored_lut_cpu_gpu_max_error=application_error,
                gpu_rerun_by_auditor=False,skin_truth_annotation_available=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
