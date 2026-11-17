"""Compare a pinned external color pipeline with the saved adaptive baseline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from probe_reference_chroma_spread import (
    BACKGROUND, BASE, GLOBAL, PARENT, decision, digest, projection, read, region_delta, target_regions,
)

REVISION = '9ef7f1cd747f542d4201297ef86d43e79493bad8'
PARENT_SHA256 = 'd7dac2996d03cdf8d5a08c3e0200272772bfd8c6f7fe0d81931c84db01cd33dc'
OUT = BASE / 'chromagrade-screen-v1'


@torch.inference_mode()
def run(upstream, out):
    git = ['git', '-C', str(upstream)]
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output([*git, 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    sys.path.insert(0, str(upstream.resolve()))
    from chromagrade import MatchParams, color_match

    torch.set_num_threads(4)
    parent_report = PARENT / 'results.json'
    assert digest(parent_report) == PARENT_SHA256
    data = json.loads(parent_report.read_text(encoding='utf-8'))
    assert data['finished'] and data['strength'] == .8 and len(data['cases']) == 6
    params = MatchParams(strength=.8)
    out.mkdir(exist_ok=False)
    protocol = Path(__file__).with_name('chromagrade-screen-protocol.md')
    report = dict(source_revision=REVISION, params=asdict(params), parent_sha256=PARENT_SHA256,
                  model_sha256=data['model_sha256'], protocol_sha256=digest(protocol),
                  script_sha256=digest(__file__), torch_version=torch.__version__, cases=[])
    start = time.monotonic()
    for case in data['cases']:
        target = read(case['input_file'])
        target_tensor = torch.from_numpy(target).float().unsqueeze(0) / 255
        assert target.shape == (256, 256, 3)
        pair = case['pair']
        folder = out / pair
        folder.mkdir()
        mask_name = pair.split('-')[0] + '-human-mask.png'
        mask_path = BASE / 'independent-coco' / mask_name
        assert digest(mask_path) == case['parents'][mask_name]
        with Image.open(mask_path) as mask_image:
            human = np.asarray(mask_image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        masks = target_regions(target, human)
        variants = {v['name']: v for v in case['variants']}
        records = {}
        for name in ['identity', *BACKGROUND, *GLOBAL]:
            variant = variants[name]
            assert not variant['certified_aligned']
            reference = torch.from_numpy(read(variant['reference'])).float().unsqueeze(0) / 255
            graded = color_match(target_tensor, reference, params)
            assert graded.device.type == 'cpu' and graded.shape == target_tensor.shape
            assert torch.isfinite(graded).all() and torch.all((graded >= 0) & (graded <= 1))
            assert time.monotonic() - start < 300, 'CPU budget exceeded'
            candidate = (graded[0] * 255).round().to(torch.uint8).numpy()
            base = read(variant['methods']['adaptive']['file'])
            path = folder / f'{name}-candidate.png'
            Image.fromarray(candidate).save(path)
            outputs = {'baseline': base, 'candidate': candidate}
            if name == 'identity':
                origin = outputs
            deltas = {key: image.astype(np.int16) - origin[key] for key, image in outputs.items()}
            records[name] = dict(
                output_sha256=digest(path),
                regions={key: {r: region_delta(image, origin[key], mask) for r, mask in masks.items()}
                         for key, image in outputs.items()},
                projection={r: projection(deltas['candidate'], deltas['baseline'], mask)
                            for r, mask in masks.items()},
                change_from_baseline={r: region_delta(candidate, base, mask) for r, mask in masks.items()},
            )
        report['cases'].append(dict(pair=pair, variants=records))
        print(pair, flush=True)
    report.update(decision=decision(report['cases']), wall_seconds=time.monotonic() - start)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['decision']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.upstream, args.out)
