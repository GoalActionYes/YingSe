"""Read-only cached-preview overhead profile; synthetic cached grade, no GPU.

This diagnoses validation/encoding cost, not grading quality or HTTP latency.
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace
import hashlib
import numpy as np
from PIL import Image
from canonical_model import ROOT
sys.path.insert(0,str(ROOT))
from spatial_store import SpatialStore
from reference_color_app.server import png


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='preview-profile-v1');args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in ('','.','..'):raise ValueError('Expected directory name')
    out=ROOT/'outputs/adaptive-reference-20261113/natural-video'/args.run;out.mkdir(exist_ok=False)
    root=ROOT/'outputs/adaptive-reference-20261113/natural-video/http-v2/data'
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'mappings').glob('*.json')]
    record=next(r for r in records if json.loads((root/'media'/(r['target_id']+'.json')).read_text(encoding='utf-8'))['metadata']['duration_seconds']>10)
    store=SpatialStore(root,SimpleNamespace(device='cpu'),identity_tag=record['identity']['spatial_engine'])
    source=store.get_media(record['target_id']);rgb,_=store.decode(source,0)
    image=Image.fromarray(rgb);image.thumbnail((960,640),Image.Resampling.BILINEAR);rgb=np.array(image)
    key=(record['id'],rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest());store.preview_cache[key]=rgb.copy()
    durations=[];profiler=cProfile.Profile();profiler.enable()
    for _ in range(10):
        start=time.perf_counter();source=store.get_media(record['target_id'])
        result=store.render(record['id'],rgb,.5,preview=True,verified_target=source);png(result)
        durations.append(time.perf_counter()-start)
    profiler.disable();stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(45)
    (out/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    report={'scope':'Direct CPU cache-hit validation and PNG encoding, synthetic cached grade; no HTTP/GPU/quality acceptance','seconds':durations}
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(stream.getvalue());print(json.dumps(report))


if __name__=='__main__':main()
