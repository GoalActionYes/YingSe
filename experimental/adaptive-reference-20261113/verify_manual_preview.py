"""Freeze broad Python finishing oracles, then check the local JS numeric kernel."""
import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from independent_coco import ROOT,digest
from spatial_render import finish_spatial
from reference_color_app.adjustments import normalize,BASIC,COLORS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='numeric-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected directory name')
    out=ROOT/'outputs/adaptive-reference-20261113/manual-preview'/args.run;out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(1406)
    before=rng.integers(0,256,(128,512,3),dtype=np.uint8);full=rng.integers(0,256,before.shape,dtype=np.uint8)
    # Include black, white, near gray, hue wraparound, primaries and clipping extremes.
    palette=np.array([[0,0,0],[255,255,255],[127,127,127],[128,127,128],[255,0,0],[0,255,0],[0,0,255],[255,0,1],[255,1,0],[1,0,0]],np.uint8)
    before[0,:len(palette)]=palette;full[0,:len(palette)]=palette[::-1]
    rgba=lambda rgb:np.concatenate([rgb,np.full((*rgb.shape[:2],1),255,np.uint8)],axis=2)
    rgba(before).tofile(out/'before.rgba');rgba(full).tofile(out/'full.rgba')
    cases=[]
    def add(name,state,strength=.37):cases.append(dict(name=name,settings=normalize(state),strength=strength))
    add('neutral',{})
    for key in BASIC:
        for value in [-100,100]:add(f'{key}:{value}',dict(basic={key:value}))
    for key in COLORS:
        for channel in range(3):
            for value in [-100,100]:
                v=[0,0,0];v[channel]=value;add(f'{key}:{channel}:{value}',dict(hsl={key:v}))
    add('basic',dict(basic=dict(temperature=24,tint=-13,brightness=16,contrast=20,vibrance=30)))
    add('hsl',dict(hsl=dict(red=[12,20,-10],blue=[-8,-15,12])))
    curves=dict(rgb=[[0,.04],[.2,.13],[.6,.72],[1,.93]],r=[[0,0],[.5,.6],[1,1]],g=[[0,0],[1,.94]],b=[[0,.04],[1,1]])
    add('curves',dict(curves=curves))
    combined=copy.deepcopy(cases[-3]['settings']);combined['hsl']=copy.deepcopy(cases[-2]['settings']['hsl']);combined['curves']=normalize(dict(curves=curves))['curves'];add('combined',combined)
    for strength in [0,.01,.5,.8,1]:add(f'combined-strength:{strength}',combined,strength)
    add('disabled',dict(basic=dict(enabled=False,brightness=80),hsl=dict(enabled=False,red=[100,100,100]),curves=dict(curves,enabled=False)))
    for i in range(16):
        state=normalize();state['basic'].update({k:int(rng.integers(-100,101)) for k in BASIC})
        state['hsl'].update({k:rng.integers(-100,101,3).tolist() for k in COLORS})
        state['curves'].update({k:[[0,float(rng.random()*.1)],[.25,float(rng.random())],[.75,float(rng.random())],[1,float(.9+rng.random()*.1)]] for k in ['rgb','r','g','b']})
        add(f'random:{i}',state,float(rng.integers(0,101))/100)
    for i,item in enumerate(cases):rgba(finish_spatial(before,full,item['strength'],item['settings'])).tofile(out/f'expected-{i}.rgba')
    (out/'cases.json').write_text(json.dumps(cases),encoding='utf-8')
    node=shutil.which('node')
    if node is None:raise RuntimeError('Node is required for developer validation')
    result=subprocess.run([node,str(Path(__file__).with_name('verify_manual_math.cjs')),str(out)],capture_output=True,timeout=180)
    (out/'node-output.txt').write_bytes(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8','replace'))
    report=json.loads(result.stdout);report['python_oracle']='finish_spatial: original/full RGB8 -> float strength blend -> existing manual adjustments -> final RGB8'
    report['python_finish_benchmarks']=[]
    a=np.tile(before,(5,2,1))[:640,:960];b=np.tile(full,(5,2,1))[:640,:960]
    for name in ['basic','hsl','curves','combined']:
        item=next(c for c in cases if c['name']==name);times=[]
        for _ in range(3):
            start=time.perf_counter();finish_spatial(a,b,item['strength'],item['settings']);times.append(time.perf_counter()-start)
        report['python_finish_benchmarks'].append(dict(name=name,dimensions=[960,640],seconds=times,scope='Python finishing only; cached synthetic grade, excludes integrity validation, model, HTTP and PNG'))
    report['source_sha256']={n:digest(Path(__file__).with_name(n)) for n in ['spatial_manual_preview.js','verify_manual_preview.py','verify_manual_math.cjs']}
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
