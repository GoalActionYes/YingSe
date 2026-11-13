"""Same-process alternating model timing; no quality or realtime claim."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine


def main():
    torch.set_num_threads(4);base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    out=base/'multiref-runtime-v1';out.mkdir(exist_ok=False)
    for name in ['benchmark_multiref.py','spatial_render.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    data=base/'cohort-multiref-v1';cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));prep=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'))
    paths={e['original_path']:e for e in prep['files']};item=cohort['groups']['validation'][0]
    def load(role):
        entry=paths[item[role]];path=data/entry['prepared_path']
        if digest(path)!=entry['sha256']:raise ValueError('Changed timing fixture')
        with Image.open(path) as im:return im.convert('RGB').copy()
    photo=load('content');reference=np.array(load('reference'))
    engines={name:SpatialEngine(name) for name in ['app8','multiref-v1']};rows=[]
    for width,height in [(640,360),(1920,1080)]:
        target=np.array(photo.resize((width,height),Image.Resampling.BILINEAR))
        for engine in engines.values():engine.prepare(target,reference).apply(target)
        for repeat in range(6):
            order=['app8','multiref-v1'] if repeat%2==0 else ['multiref-v1','app8']
            for name in order:
                start=time.perf_counter();prepared=engines[name].prepare(target,reference);prepared.apply(target);seconds=time.perf_counter()-start
                rows.append(dict(profile=name,repeat=repeat,width=width,height=height,seconds=seconds))
    summary={}
    for width,height in [(640,360),(1920,1080)]:
        summary[f'{width}x{height}']={}
        for name in engines:
            values=[r['seconds'] for r in rows if r['profile']==name and r['width']==width]
            summary[f'{width}x{height}'][name]=dict(median=float(np.median(values)),minimum=float(min(values)),maximum=float(max(values)))
    report=dict(scope='Warm alternating same-process resized-photo timing, not complete video or UI latency',
        input_sha256={role:paths[item[role]]['sha256'] for role in ['content','reference']},torch_version=torch.__version__,device=torch.cuda.get_device_name(),
        source_sha256={name:digest(out/name) for name in ['benchmark_multiref.py','spatial_render.py']},checkpoint_sha256={name:e.checkpoint_sha for name,e in engines.items()},summary=summary,rows=rows)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
