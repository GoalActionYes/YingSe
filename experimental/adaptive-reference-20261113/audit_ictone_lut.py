"""Offline paired scores and independent CPU application of stored author LUTs."""
import argparse
import json
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
    out=base/args.run
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    result=load(out/'results.json')
    assert result['finished'] and result['author_sha256']==digest(ROOT/'.tools/ICTone-audit/inference.py')
    assert result['source_sha256']==digest(out/'reconstruct_ictone_lut.py')
    assert digest(out/'protocol-frozen.md')==result['protocol_sha256']
    screen=base/result['screen']
    assert digest(screen/'results.json')==result['screen_result_sha256']
    raw=load(screen/'results.json')
    assert [x['index'] for x in result['rows']]==[x['index'] for x in raw['completed']]
    funcs=author_functions()
    data=base/'cohort-training-coverage-v2'
    rows={x['index']:x for x in load(data/'cohort.json')['groups']['validation']}
    lookup={x['original_path']:x for x in load(data/'srgb-v1/manifest.json')['files']}
    panel=Image.new('RGB',(1536,282*len(result['rows'])),'white')
    draw=ImageDraw.Draw(panel)
    measurements=[]
    max_cpu_gpu=0
    for ordinal,(item,previous) in enumerate(zip(result['rows'],raw['completed'])):
        index=item['index']
        folder=base/item['source_run']
        assert item['source_run']==previous['run']
        for name,key in [('content256.png','content_sha256'),('native512.png','native_sha256')]:
            assert digest(folder/name)==item[key]
        assert digest(out/(str(index)+'-lut.npy'))==item['lut_sha256']
        assert digest(out/(str(index)+'.png'))==item['output_sha256']
        lut=np.load(out/(str(index)+'-lut.npy'),allow_pickle=False)
        assert lut.shape==(33,33,33,3) and np.isfinite(lut).all()
        with Image.open(folder/'content256.png') as im:content=im.copy()
        with Image.open(out/(str(index)+'.png')) as im:projection=im.copy()
        portable=funcs['apply_lut'](np.array(content),lut,device='cpu')
        error=int(np.abs(portable.astype(int)-np.array(projection).astype(int)).max())
        assert error<=1,error
        max_cpu_gpu=max(max_cpu_gpu,error)
        gt=lookup[rows[index]['gt']]
        gtpath=data/gt['prepared_path']
        assert gtpath.resolve().is_relative_to(base.resolve()) and digest(gtpath)==gt['sha256']
        with Image.open(gtpath) as im:truth=im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        with Image.open(base/'multiref-control-v1/epoch-6'/f'{index}.png') as im:baseline=im.copy()
        with Image.open(folder/'metric256.png') as im:original=im.copy()
        with Image.open(folder/'reference256.png') as im:reference=im.copy()
        t=np.array(truth).astype(float)
        old_mae=float(np.abs(np.array(baseline).astype(float)-t).mean())
        raw_mae=float(np.abs(np.array(original).astype(float)-t).mean())
        mae=float(np.abs(np.array(projection).astype(float)-t).mean())
        assert old_mae==previous['baseline_mae'] and raw_mae==previous['candidate_mae']
        measurements.append(dict(index=index,baseline_mae=old_mae,raw_mae=raw_mae,lut_mae=mae,
                                 severe=bool(mae-old_mae>5 and mae>old_mae*1.25),
                                 lut_vs_raw_mae=float(np.abs(np.array(projection).astype(float)-np.array(original).astype(float)).mean())))
        for column,(label,picture) in enumerate([('Content',content),('Reference',reference),('GT offline',truth),
                ('current13',baseline),('Raw Q3',original),('Author LUT33',projection)]):
            top=ordinal*282
            panel.paste(picture,(256*column,top+26))
            draw.text((256*column+5,top+7),str(index)+' '+label,fill='black')
    panel.save(out/'diagnostic-panel.png')
    audit=dict(finished=True,scope='completed consumed subset; no independent/human/product acceptance',
               result_sha256=digest(out/'results.json'),source_sha256=digest(Path(__file__)),rows=measurements,
               saved_lut_application_cpu_gpu_max_error=max_cpu_gpu,
               independent_solver_rerun=False,solver_convergence_metadata_exposed_by_author=False,
               baseline_mean=sum(x['baseline_mae'] for x in measurements)/len(measurements),
               raw_mean=sum(x['raw_mae'] for x in measurements)/len(measurements),
               lut_mean=sum(x['lut_mae'] for x in measurements)/len(measurements),
               improvement_count=sum(x['baseline_mae']-x['lut_mae']>.1 for x in measurements),
               severe_count=sum(x['severe'] for x in measurements))
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
