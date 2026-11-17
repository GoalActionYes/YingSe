"""Screen the frozen expanded checkpoint on the six existing background controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_pairwise import appearance_forward
from appearance_replication import initialize_replication_model
from probe_reference_chroma_spread import (
    BACKGROUND, BASE, GLOBAL, PARENT, decision, digest, projection, read, region_delta, target_regions,
)
from spatial_render import SpatialEngine, finish_spatial

WEIGHT = BASE / 'tst100k/training-coverage-expanded-v2/epoch-4.pth'
WEIGHT_SHA256 = 'cde15ae2d5295de709b14c8a4fe6e72e3cd7343fa5766d6ac30a9c8858db3ee1'
PARENT_SHA256 = 'd7dac2996d03cdf8d5a08c3e0200272772bfd8c6f7fe0d81931c84db01cd33dc'


@torch.inference_mode()
def run(out):
    assert digest(WEIGHT) == WEIGHT_SHA256
    parent_report = PARENT / 'results.json'
    assert digest(parent_report) == PARENT_SHA256
    parent = json.loads(parent_report.read_text(encoding='utf-8'))
    assert parent['finished'] and parent['strength'] == .8 and len(parent['cases']) == 6
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == parent['model_sha256']
    initialize_replication_model(engine.model, engine.model.state_dict())
    state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    assert all(torch.isfinite(tensor).all() for tensor in state.values())
    engine.model.load_state_dict(state, strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir(exist_ok=False)
    protocol = Path(__file__).with_name('expanded-background-protocol.md')
    report = dict(parent_sha256=PARENT_SHA256, candidate_checkpoint_sha256=WEIGHT_SHA256,
                  baseline_checkpoint_sha256=engine.checkpoint_sha, script_sha256=digest(__file__),
                  protocol_sha256=digest(protocol), cases=[],
                  appearance_source_sha256=digest(Path(__file__).with_name('appearance_pairwise.py')),
                  initializer_sha256=digest(Path(__file__).with_name('appearance_replication.py')))
    start = time.monotonic()
    for case in parent['cases']:
        pair = case['pair']
        folder = out / pair
        folder.mkdir()
        target = read(case['input_file'])
        target_tensor = engine.analysis(target)
        person_file = pair.split('-')[0] + '-human-mask.png'
        person_path = BASE / 'independent-coco' / person_file
        assert digest(person_path) == case['parents'][person_file]
        with Image.open(person_path) as mask_image:
            person = np.asarray(mask_image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        masks = target_regions(target, person)
        variants = {row['name']: row for row in case['variants']}
        records = {}
        for name in ['identity', *BACKGROUND, *GLOBAL]:
            row = variants[name]
            assert not row['certified_aligned']
            baseline = read(row['methods']['adaptive']['file'])
            reference_tensor = engine.analysis(read(row['reference']))
            raw, _ = appearance_forward(engine.model, engine.api, target_tensor, reference_tensor)
            full = np.rint(raw[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            candidate = finish_spatial(target, full, .8)
            path = folder / f'{name}-candidate.png'
            Image.fromarray(candidate).save(path)
            outputs = {'baseline': baseline, 'candidate': candidate}
            if name == 'identity':
                origin = outputs
            changes = {key: image.astype(np.int16) - origin[key] for key, image in outputs.items()}
            records[name] = dict(
                output_sha256=digest(path),
                regions={key: {r: region_delta(image, origin[key], mask) for r, mask in masks.items()}
                         for key, image in outputs.items()},
                projection={r: projection(changes['candidate'], changes['baseline'], mask)
                            for r, mask in masks.items()},
                change_from_baseline={r: region_delta(candidate, baseline, mask) for r, mask in masks.items()},
            )
            assert time.monotonic() - start < 180
        report['cases'].append(dict(pair=pair, variants=records))
        print(pair, flush=True)
    report.update(decision=decision(report['cases']), seconds=time.monotonic()-start)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['decision']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
