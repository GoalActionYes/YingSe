"""Isolated research invocation of the author's separately licensed smoother."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    run=Path(sys.argv[1]).resolve()
    allowed=(ROOT/'outputs/adaptive-reference-20261113/pca-kd').resolve()
    if not run.is_relative_to(allowed):raise ValueError('Unexpected research run directory')
    source=ROOT/'.tools/PCA-KD-source'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    relative='VGG backbone/utils/photo_gif.py'
    expected=next(x['sha256'] for x in manifest['files'] if x['path']==relative)
    if digest(source/relative)!=expected:raise ValueError('Smoother source changed')
    spec=importlib.util.spec_from_file_location('author_smoother',source/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    smoother=module.GIFSmoothing(30,(.02*255)**2)
    timings={}
    with tempfile.TemporaryDirectory(prefix='yingse-pca-guide-') as temp:
        guide=Path(temp)/'guide.png'
        for file in sorted(run.glob('*-raw.npz')):
            with np.load(file,allow_pickle=False) as data:
                Image.fromarray(data['guide']).save(guide)
                start=time.perf_counter();result=smoother.process(data['rgb'],str(guide))
                timings[file.stem]=time.perf_counter()-start
                result.save(run/(file.stem.replace('-raw','-guided')+'.png'))
    (run/'smoothing.json').write_text(json.dumps(dict(timing_seconds=timings,source_sha256=expected,
        license='Author photo_gif.py NVIDIA CC BY-NC-SA 4.0; research invocation only')),encoding='utf-8')


if __name__=='__main__':main()
