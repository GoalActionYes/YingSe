"""Compare one research-only target-geometry change at fixed color conditioning."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from probe_playraw_mixed_light_transfer import (
    BASE, CHECKPOINT, PARENT, PARENT_SHA, STRENGTH, chroma, read, stats,
)
from probe_psbench_teal_portrait import IDS, verify_source
from probe_reference_chroma_spread import projection
from probe_target_mirror_chroma import checked_json, digest, json_to, save
from reference_color_app.media import decode_image
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from training_checks import finite_output


@torch.inference_mode()
def render(engine, target, reference, preserve_geometry=False):
    height, width = target.shape[:2]
    if np.array_equal(target, reference):
        return target.copy()
    if not preserve_geometry or height == width:
        return finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)

    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    padded = np.pad(target, ((top, side - height - top), (left, side - width - left), (0, 0)), mode='reflect')
    api = engine.api
    source = engine.analysis(target)
    look = engine.analysis(reference)
    hc = api['batch_histogram_1D'](api['srgb_to_ycbcr'](source))
    hs = api['batch_histogram_1D'](api['srgb_to_ycbcr'](look))
    low = api['TF'].resize(engine.analysis(padded), (64, 64), antialias=True)
    grid = api['lowpass'](engine.model.apply_grid(low, torch.cat([hc, hs], 1)))
    finite_output(grid, (1, 13, 8, 16, 16))
    rendered = PreparedSpatial(engine, grid, {}).apply(padded)
    return finish_spatial(target, rendered[top:top + height, left:left + width], STRENGTH)


ARMS = {'stretched': False, 'padded': True}


def psbench(engine, out):
    folder = BASE / 'psbench-shared-local-v1'
    evidence_dir = Path(__file__).resolve().parents[2] / 'docs/research'
    native = json.loads((evidence_dir / '2026-11-18-psbench-native-evidence.json').read_bytes())
    rows = {}
    for task, reference_task in (IDS, IDS[::-1]):
        assert verify_source(folder, task) == native['files'][task]
        target = decode_image(folder / f'{task}_start.jpg')
        reference = decode_image(folder / f'{reference_task}_end.jpg')
        grades = {arm: render(engine, target, reference, variant) for arm, variant in ARMS.items()}
        outputs = {arm: save(out / f'psbench-{task}-{arm}.png', grade) for arm, grade in grades.items()}
        assert outputs['stretched'] == native['predictions'][task]
        expected = finish_spatial(target, decode_image(folder / f'{task}_end.jpg'), STRENGTH).astype(np.int16)
        desired = expected - target
        regions = target_regions(target, np.zeros(target.shape[:2], bool))
        scores = {
            arm: {name: {'mae': float(np.abs((grade.astype(np.int16) - expected)[regions[name]]).mean()),
                         'projection': projection(grade.astype(np.int16) - target, desired, regions[name])}
                  for name in ('whole', 'neutral_proxy')}
            for arm, grade in grades.items()
        }
        baseline, candidate = scores['stretched'], scores['padded']
        passed = (baseline['neutral_proxy']['mae'] - candidate['neutral_proxy']['mae'] >= 1
                  and candidate['whole']['mae'] <= baseline['whole']['mae'] + 1
                  and candidate['whole']['projection'] >= baseline['whole']['projection'])
        rows[task] = {'size_wh': target.shape[:2][::-1], 'sha256': outputs, 'score': scores, 'passed': passed}
    return rows


def playraw(engine, out):
    recorded = checked_json(PARENT / 'results.json', PARENT_SHA)
    target, _ = read(PARENT / 'off.tif', False)
    edited, mask = read(PARENT / 'on.tif', True)
    assert [sha256(rgb.tobytes()).hexdigest() for rgb in (target, edited)] == recorded['rgb_sha256']
    assert sha256(mask.tobytes()).hexdigest() == recorded['mask_pixels_sha256']
    references = [np.ascontiguousarray(image[:, ::-1]) for image in (target, edited)]
    grades = {arm: [render(engine, target, ref, variant) for ref in references] for arm, variant in ARMS.items()}
    names = ('off', 'on')
    outputs = {arm: {name: save(out / f'playraw-{arm}-{name}.png', grade)
                     for name, grade in zip(names, images, strict=True)} for arm, images in grades.items()}
    assert outputs['stretched'] == {
        'off': '1932e7069e9802466034a751a9ddf18ff1b9661c5177718ea2e45f8664a97a67',
        'on': '4e41104df1c8bf300b13d22d203d6f70ea4cfe3bf73561d27e22e5a770ed381c',
    }
    expected = chroma(finish_spatial(target, edited, STRENGTH)) - chroma(target)
    neutral = target_regions(target, np.zeros(mask.shape, bool))['neutral_proxy']
    zero = np.zeros_like(expected)
    zones = {'author': (mask >= 204, expected), 'protected': (mask == 0, zero),
             'neutral': (neutral & (mask == 0), zero)}
    changes = {arm: chroma(on) - chroma(off) for arm, (off, on) in grades.items()}
    scores = {arm: {name: stats(change, truth, region) for name, (region, truth) in zones.items()}
              for arm, change in changes.items()}
    candidate = scores['padded']
    passed = (.75 <= candidate['author']['projection'] <= 1.25 and candidate['author']['response_mae'] <= 3
              and max(candidate[name]['response_p95'] for name in ('protected', 'neutral')) <= 1.5)
    square = np.asarray(Image.fromarray(target).resize((256, 256), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(render(engine, square, references[1]), render(engine, square, references[1], True))
    np.testing.assert_array_equal(render(engine, target, target.copy(), True), target)
    return {'size_wh': target.shape[:2][::-1], 'sha256': outputs, 'score': scores, 'passed': passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    out = parser.parse_args().out
    out.mkdir()
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == CHECKPOINT
    cross, local = psbench(engine, out), playraw(engine, out)
    json_to(out / 'results.json', {
        'checkpoint_sha256': CHECKPOINT, 'source_sha256': digest(Path(__file__)),
        'protocol_sha256': digest(Path(__file__).with_name('target-geometry-protocol.md')),
        'strength': STRENGTH, 'square_exact': True, 'identity_exact': True, 'psbench': cross,
        'playraw': local, 'passed': all(row['passed'] for row in cross.values()) and local['passed'],
    })


if __name__ == '__main__':
    main()
