"""Fixed first-eight screen after exact disk-reuse bridge; GT stays in parent."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from screen_ictone_first import ROOT, digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--bridge',default='ictone-reuse-bridge-v1')
    args=p.parse_args()
    assert all(Path(x).name==x and x not in ('.','..') for x in [args.run,args.bridge])
    import numpy as np
    import psutil
    from PIL import Image
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    old=base/'ictone-first-v1'
    bridge=base/args.bridge
    original,producer,audit=load(old/'results.json'),load(bridge/'results.json'),load(bridge/'audit.json')
    assert producer['finished'] and audit['finished'] and audit['state_parameters_verified']==1846
    assert producer['reused_stage_from']=='ictone-first-v1' and producer['config']==original['config']
    assert producer['inputs']==original['inputs'] and producer['checked_parameter_count']==1846
    assert audit['producer_result_sha256']==digest(bridge/'results.json')
    for name,sha in producer['source_sha256'].items():
        assert digest(Path(__file__).with_name(name))==sha==digest(bridge/name)
    exact={}
    for name in ['triptych-output.png','native512.png','metric256.png']:
        with Image.open(old/name) as im:a=np.array(im)
        with Image.open(bridge/name) as im:b=np.array(im)
        exact[name]=bool(np.array_equal(a,b))
    assert all(exact.values()), 'bridge failed; no prior/new image results may be merged'
    data=base/'cohort-training-coverage-v2'
    assert digest(data/'cohort.json')=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    assert digest(data/'srgb-v1/manifest.json')=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    rows=load(data/'cohort.json')['groups']['validation'][:8]
    assert [x['index'] for x in rows]==[22927,88391,39532,50960,23001,3026,10960,25373]
    lookup={x['original_path']:x for x in load(data/'srgb-v1/manifest.json')['files']}
    baseline={x['index']:x for x in load(base/'training-coverage-expanded-v2/results.json')['baseline']['cases']}
    out=base/args.run
    out.mkdir(exist_ok=False)
    report=dict(finished=False,scope='consumed first-eight offline research; no independent or product acceptance',
                config=producer['config'],order=[x['index'] for x in rows],completed=[],
                bridge=dict(run=args.bridge,pixels_exact=exact,producer_sha256=digest(bridge/'results.json'),
                            audit_sha256=digest(bridge/'audit.json'),separately_produced_wall_seconds=producer['elapsed_seconds']),
                source_sha256={n:digest(Path(__file__).with_name(n)) for n in
                               ['screen_ictone_development.py','screen_ictone_first.py','ictone_flux_stage.py','ictone_gguf_runtime.py']},
                protocol_sha256=digest(ROOT/'docs/research/2026-11-17-ictone-screen.md'),
                training_overlap_not_excluded=True,product_integrated=False,human_evaluations=0)
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    started=time.monotonic()
    try:
        for ordinal,row in enumerate(rows):
            assert time.monotonic()-started<=2400, '2400-second parent budget exhausted'
            index=row['index']
            run=args.bridge if ordinal==0 else args.run+'-'+str(index)
            folder=base/run
            if ordinal:
                assert not folder.exists(), 'do not overwrite or implicitly reuse an earlier child'
                env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
                with (out/(str(index)+'-child.log')).open('w',encoding='utf-8') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('screen_ictone_first.py')),
                        '--run',run,'--index',str(ordinal),'--reuse-stage','ictone-first-v1'],
                        stdout=log,stderr=subprocess.STDOUT,env=env)
                    try:
                        code=child.wait(timeout=min(660,2400-(time.monotonic()-started)))
                    except subprocess.TimeoutExpired:
                        owned=psutil.Process(child.pid)
                        for descendant in owned.children(recursive=True):
                            descendant.kill()
                        child.kill()
                        child.wait()
                        raise RuntimeError('child execution budget exhausted; owned process tree stopped') from None
                assert code==0,(index,code)
            result=load(folder/'results.json')
            assert result['finished'] and result['completed_steps']==4 and result['index']==index
            assert result['checked_parameter_count']==1846 and result['config']==producer['config']
            assert result['source_sha256']==producer['source_sha256']
            assert result['reused_stage_from']=='ictone-first-v1'
            for name,sha in result['output_sha256'].items():assert digest(folder/name)==sha
            item=lookup[row['gt']]
            path=data/item['prepared_path']
            assert path.resolve().is_relative_to(base.resolve()) and digest(path)==item['sha256']
            with Image.open(path) as im:truth=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).astype(float)
            oldpath=base/'multiref-control-v1/epoch-6'/f'{index}.png'
            assert digest(oldpath)==baseline[index]['sha256']
            with Image.open(oldpath) as im:old_pixels=np.array(im).astype(float)
            with Image.open(folder/'metric256.png') as im:pred=np.array(im).astype(float)
            baseline_mae=float(np.abs(old_pixels-truth).mean())
            assert baseline_mae==baseline[index]['mae']
            mae=float(np.abs(pred-truth).mean())
            severe=bool(mae-baseline_mae>5 and mae>baseline_mae*1.25)
            report['completed'].append(dict(index=index,run=run,baseline_mae=baseline_mae,candidate_mae=mae,
                                            severe=severe,producer_sha256=digest(folder/'results.json'),
                                            metric_sha256=digest(folder/'metric256.png'),
                                            image_pipeline_seconds=result['image_pipeline_seconds']))
            report.update(parent_wall_seconds=time.monotonic()-started,
                          severe_count=sum(x['severe'] for x in report['completed']))
            print(json.dumps(report['completed'][-1]),flush=True)
            save()
            if report['severe_count']>=3:
                report['stop_reason']='necessary_severe_gate'
                break
        else:
            report['stop_reason']='all_eight_completed'
        report.update(finished=True,all_eight_completed=len(report['completed'])==8,
                      full64_severe_gate_impossible=report['severe_count']>=3,
                      parent_wall_seconds=time.monotonic()-started)
    except Exception as error:
        report.update(stop_reason='execution_failure',error=repr(error),parent_wall_seconds=time.monotonic()-started)
        save()
        raise
    save()


if __name__=='__main__':
    main()
