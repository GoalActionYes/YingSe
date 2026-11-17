"""Six fixed consumed COCO controls, RGB-only child and parent-only truth."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ictone_flux_stage import ROOT
from screen_ictone_first import digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('.','..')
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from reconstruct_ictone_lut import author_functions
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    inputs=base/'ictone-coco-inputs-v1'
    plan=json.loads((inputs/'plan.json').read_text())
    assert plan['finished'] and plan['baseline_bridge_exact'] and len(plan['cases'])==6
    assert [(x['target'],x['profile']) for x in plan['cases']]==[(785,'same_image'),(9483,'global_contrast'),
            (872,'global_desaturate'),(8021,'region_color_luminance'),(285,'region_tone_saturation'),(2149,'same_image')]
    out=base/args.run
    out.mkdir(exist_ok=False)
    protocol=ROOT/'docs/research/2026-11-17-ictone-coco.md'
    assert digest(protocol)==plan['protocol_sha256']
    (out/'protocol-frozen.md').write_bytes(protocol.read_bytes())
    funcs=author_functions()
    report=dict(finished=False,scope='cross-source consumed development diagnosis, not independent or product acceptance',
                plan_sha256=digest(inputs/'plan.json'),protocol_sha256=digest(protocol),completed=[],
                input_schema='only index and two RGB files/hashes; rejects GT/masks/profile',
                source_sha256={name:digest(Path(__file__).with_name(name)) for name in
                    ['screen_ictone_coco.py','screen_ictone_first.py','ictone_input_spec.py','ictone_flux_stage.py',
                     'ictone_gguf_runtime.py','reconstruct_ictone_lut.py']})
    for name in report['source_sha256']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    started=time.monotonic()
    def read(path,resampling=Image.Resampling.BILINEAR):
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),resampling))
    try:
        for item in plan['cases']:
            remaining=1800-(time.monotonic()-started)
            assert remaining>0,'1800-second parent budget exhausted'
            assert digest(Path(item['spec_path']))==item['spec_sha256']
            run=args.run+'-'+item['index']
            folder=base/run
            assert not folder.exists()
            env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
            with (out/(item['index']+'-child.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('screen_ictone_first.py')),
                    '--run',run,'--reuse-stage','ictone-first-v1','--input-spec',item['spec_path']],
                    stdout=log,stderr=subprocess.STDOUT,env=env)
                try:code=child.wait(timeout=min(660,remaining))
                except subprocess.TimeoutExpired:
                    for descendant in psutil.Process(child.pid).children(recursive=True):descendant.kill()
                    child.kill();child.wait()
                    raise RuntimeError('owned child tree stopped after execution budget') from None
            assert code==0,(item['index'],code)
            result=json.loads((folder/'results.json').read_text())
            assert result['finished'] and result['index']==item['index'] and result['checked_parameter_count']==1846
            assert result['input_spec_sha256']==item['spec_sha256']
            for name,sha in result['source_sha256'].items():assert sha==report['source_sha256'][name]
            spec=json.loads(Path(item['spec_path']).read_text())
            for role in ['content','reference']:
                assert digest(folder/(role+'256.png'))==spec[role]['sha256']
            content=read(folder/'content256.png')
            after=read(folder/'native512.png')
            then=time.monotonic()
            lut=funcs['estimate_lut'](content.reshape(-1,3),after.reshape(-1,3),size=33,device='cuda')
            assert np.isfinite(lut).all()
            projected=funcs['apply_lut'](content,lut,device='cuda')
            torch.cuda.synchronize()
            reconstruct_seconds=time.monotonic()-then
            Image.fromarray(projected).save(out/(item['index']+'.png'))
            np.save(out/(item['index']+'-lut.npy'),lut)
            # Only now open GT and manual mask for offline measurement.
            assert digest(Path(item['truth_source']))==item['truth_sha256']
            truth=read(item['truth_source']).astype(float)
            assert digest(Path(item['baseline_path']))==item['baseline_sha256']
            baseline=read(item['baseline_path'])
            with Image.open(item['human_mask_source']) as im:
                mask=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
            def score(rgb):
                err=np.abs(rgb.astype(float)-truth).mean(2)
                return dict(mae=float(err.mean()),person_mae=float(err[mask].mean()) if mask.any() else None,
                            background_mae=float(err[~mask].mean()) if (~mask).any() else None,
                            clipped_channel_fraction=float(((rgb==0)|(rgb==255)).mean()))
            old,raw,new=score(baseline),score(after),score(projected)
            severe=bool(new['mae']-old['mae']>5 and new['mae']>old['mae']*1.25)
            record=dict(index=item['index'],target=item['target'],reference=item['reference'],profile=item['profile'],
                        child_run=run,producer_sha256=digest(folder/'results.json'),baseline=old,raw=raw,candidate=new,
                        severe=severe,output_sha256=digest(out/(item['index']+'.png')),
                        lut_sha256=digest(out/(item['index']+'-lut.npy')),
                        image_pipeline_seconds=result['image_pipeline_seconds'],reconstruction_seconds=reconstruct_seconds)
            report['completed'].append(record)
            report.update(severe_count=sum(x['severe'] for x in report['completed']),parent_wall_seconds=time.monotonic()-started)
            print(json.dumps(record),flush=True)
            save()
            torch.cuda.empty_cache()
            if report['severe_count']>=3:
                report['stop_reason']='necessary_severe_gate'
                break
        else:report['stop_reason']='all_six_completed'
        report.update(finished=True,all_six_completed=len(report['completed'])==6,parent_wall_seconds=time.monotonic()-started)
    except Exception as error:
        report.update(stop_reason='execution_failure',error=repr(error),parent_wall_seconds=time.monotonic()-started)
        save()
        raise
    save()


if __name__=='__main__':main()
