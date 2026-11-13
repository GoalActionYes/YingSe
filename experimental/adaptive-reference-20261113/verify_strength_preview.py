"""Cross-check browser arithmetic and served-script races against Python."""
import json
import argparse
import os
import re
from pathlib import Path
import shutil
import subprocess
import numpy as np
from independent_coco import ROOT,digest
from spatial_render import finish_spatial
from spatial_workbench import assets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='numeric-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected a single output directory name')
    out=ROOT/'outputs/adaptive-reference-20261113/strength-preview'/args.run;out.mkdir(parents=True,exist_ok=False)
    node=shutil.which('node')
    if node is None:raise RuntimeError('Node is required for these developer checks')
    a,b=np.meshgrid(np.arange(256,dtype=np.uint8),np.arange(256,dtype=np.uint8),indexing='ij')
    before=np.repeat(a[...,None],3,axis=2);full=np.repeat(b[...,None],3,axis=2)
    def rgba(rgb):return np.concatenate([rgb,np.full((*rgb.shape[:2],1),255,np.uint8)],axis=2)
    rgba(before).tofile(out/'before.rgba');rgba(full).tofile(out/'full.rgba')
    with (out/'expected.rgba').open('wb') as stream:
        for strength in range(101):stream.write(rgba(finish_spatial(before,full,strength/100)).tobytes())
    (out/'golden.json').write_text(json.dumps(dict(strengths=101,pixels_per_image=65536,scope='Synthetic complete byte-pair mixing; not model quality')),encoding='utf-8')
    (out/'app.js').write_bytes(assets(True)['/app.js'][0])
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    result=subprocess.run([node,'--test','--test-reporter=tap',str(Path(__file__).with_name('test_strength_preview.cjs'))],env=dict(os.environ,SPATIAL_APP_JS=str(out/'app.js')),capture_output=True,timeout=60,creationflags=flags)
    (out/'node-tests.txt').write_bytes(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout.decode('utf-8','replace')+result.stderr.decode('utf-8','replace'))
    state_tests=int(re.search(rb'^# tests (\d+)$',result.stdout,re.M)[1])
    harness=ROOT/'tests/reference_color/test_ui.cjs'
    fallback=subprocess.run([node,'--test','--test-reporter=tap',str(harness)],env=dict(os.environ,YINGSE_APP_JS=str(out/'app.js')),capture_output=True,timeout=60,creationflags=flags)
    (out/'fallback-tests.txt').write_bytes(fallback.stdout+fallback.stderr)
    if fallback.returncode:raise RuntimeError(fallback.stdout.decode('utf-8','replace')+fallback.stderr.decode('utf-8','replace'))
    fallback_tests=int(re.search(rb'^# tests (\d+)$',fallback.stdout,re.M)[1])
    result=subprocess.run([node,str(Path(__file__).with_name('verify_strength_math.cjs')),str(out)],capture_output=True,timeout=60,creationflags=flags)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8','replace'))
    report=dict(math=json.loads(result.stdout),passed=True,browser_visual_tested=False,state_tests=state_tests,fallback_tests=fallback_tests,harness_sha256=digest(harness),
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['spatial_strength_preview.js','spatial_workbench.py','verify_strength_preview.py','verify_strength_math.cjs','test_strength_preview.cjs']})
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
