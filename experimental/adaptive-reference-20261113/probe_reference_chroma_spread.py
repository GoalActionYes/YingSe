"""Test one reference histogram weighting on the six consumed v2 cases."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from background_intervention import region_delta, target_regions
from independent_coco import ROOT, digest
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial

BASE = ROOT / 'outputs/adaptive-reference-20261113'
PARENT = BASE / 'background-intervention-20261116-v2'
OUT = BASE / 'reference-chroma-spread-v1'
BACKGROUND = ['background_green_100', 'background_magenta_100']
GLOBAL = ['whole_green', 'whole_magenta', 'whole_contrast', 'whole_desaturate']


def spread_histogram(ycbcr, original):
    chroma = ((ycbcr[0, 1:] + .5) * 128).floor().clamp(0, 127).long().cpu().numpy()
    tiles = chroma.reshape(2, 4, 64, 4, 64).transpose(0, 1, 3, 2, 4).reshape(2, 16, -1)
    counts = np.array([[np.bincount(tile, minlength=128) for tile in channel] for channel in tiles], dtype=np.float64)
    mass = counts.sum(axis=1)
    np.testing.assert_allclose(mass / 65536, original[0, 1:].cpu().numpy(), atol=1e-7, rtol=0)
    squared_mass = (counts ** 2).sum(axis=1)
    coverage = np.divide(mass ** 2, squared_mass * 16, out=np.zeros_like(mass), where=squared_mass > 0)
    weighted = mass * (.5 + .5 * coverage)
    weighted /= weighted.sum(axis=1, keepdims=True)
    result = original.clone()
    result[0, 1:] = torch.as_tensor(weighted, device=original.device, dtype=original.dtype)
    return result


def projection(candidate, base, mask):
    x, y = candidate[mask].astype(np.float64), base[mask].astype(np.float64)
    denominator = float(np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator else None


def read(entry):
    path = PARENT / entry['path']
    assert digest(path) == entry['sha256']
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def decision(cases):
    person_checks = []
    neutral_pairs = []
    whole_checks = []
    for case in cases:
        rows = case['variants']
        for background, overall in zip(BACKGROUND, GLOBAL[:2], strict=True):
            scores = rows[background]['regions']
            base, candidate = scores['baseline']['person'], scores['candidate']['person']
            neutral_pairs.append([scores[k]['neutral_proxy']['rgb_mae'] for k in ('baseline', 'candidate')])
            if base is not None:
                x, y = base['rgb_mae'], candidate['rgb_mae']
                p = rows[overall]['projection']['person']
                person_checks.append(y <= .8 * x and x - y >= .25 and p is not None and p >= .90)
        whole_checks.extend(p is not None and .90 <= p <= 1.10
                            for name in GLOBAL[2:] for p in [rows[name]['projection']['whole']])
    neutral = np.asarray(neutral_pairs)
    neutral_gain = 1 - neutral[:, 1].mean() / neutral[:, 0].mean()
    return dict(person_pass=sum(person_checks), person_total=len(person_checks),
                neutral_gain=float(neutral_gain), neutral_max_increase=float(np.max(neutral[:, 1] - neutral[:, 0])),
                whole_pass=sum(whole_checks), whole_total=len(whole_checks),
                proceed=bool(sum(person_checks) >= 6 and neutral_gain >= .15
                             and np.max(neutral[:, 1] - neutral[:, 0]) <= 1 and all(whole_checks)))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    data = json.loads((PARENT / 'results.json').read_text())
    assert data['finished'] and len(data['cases']) == 6 and data['strength'] == .8
    OUT.mkdir(exist_ok=False)
    protocol = Path(__file__).with_name('reference-chroma-spread-protocol.md')
    report = dict(parent_sha256=digest(PARENT / 'results.json'), protocol_sha256=digest(protocol),
                  source_sha256=digest(__file__), cases=[], finished=False)
    (OUT / 'config.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == data['model_sha256']
    report['model_sha256'] = engine.checkpoint_sha
    start = time.monotonic()
    for case in data['cases']:
        target = read(case['input_file'])
        pair = case['pair']
        folder = OUT / pair
        folder.mkdir()
        mask_name = pair.split('-')[0] + '-human-mask.png'
        mask_path = BASE / 'independent-coco' / mask_name
        assert digest(mask_path) == case['parents'][mask_name]
        with Image.open(mask_path) as image:
            human = np.asarray(image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        regions = target_regions(target, human)
        content = engine.analysis(target)
        color = engine.api['srgb_to_ycbcr']
        histogram = engine.api['batch_histogram_1D']
        target_histogram = histogram(color(content))
        low = engine.api['TF'].resize(content, (64, 64), antialias=True)
        inputs = {v['name']: v for v in case['variants']}
        origin = {}
        records = {}
        for name in ['identity', *BACKGROUND, *GLOBAL]:
            assert time.monotonic() - start < 180
            variant = inputs[name]
            assert not variant['certified_aligned']
            ref_color = color(engine.analysis(read(variant['reference'])))
            base_histogram = histogram(ref_color)
            outputs, hashes = {}, {}
            for label, condition in [('baseline', base_histogram), ('candidate', spread_histogram(ref_color, base_histogram))]:
                grid = engine.api['lowpass'](engine.model.apply_grid(low, torch.cat([target_histogram, condition], 1)))
                assert torch.isfinite(grid).all()
                output = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target, 1), .8)
                if label == 'baseline':
                    assert np.array_equal(output, read(variant['methods']['adaptive']['file'])), (pair, name)
                path = folder / f'{name}-{label}.png'
                Image.fromarray(output).save(path)
                outputs[label], hashes[label] = output, digest(path)
            if name == 'identity':
                origin = outputs
            deltas = {k: outputs[k].astype(np.int16) - origin[k] for k in outputs}
            records[name] = dict(output_sha256=hashes,
                                 regions={k: {r: region_delta(v, origin[k], m) for r, m in regions.items()}
                                          for k, v in outputs.items()},
                                 projection={r: projection(deltas['candidate'], deltas['baseline'], m)
                                             for r, m in regions.items()},
                                 change_from_baseline={r: region_delta(outputs['candidate'], outputs['baseline'], m)
                                                       for r, m in regions.items()})
        report['cases'].append(dict(pair=pair, variants=records))
        print(pair, flush=True)
    report.update(finished=True, decision=decision(report['cases']))
    (OUT / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['decision']))


if __name__ == '__main__':
    main()
