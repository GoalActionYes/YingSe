"""Render from separate reference halves with fixed screen-coordinate correspondence."""
import argparse
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_appearance_layout_response import LAYOUT_SHA, PARENT_SHA, SOURCE_SHA, verified
from probe_arri_shared_look import metrics
from probe_reference_layout_collision import panel as layout_panel
from probe_target_mirror_chroma import (
    BASE, BG_SHA, CHECKPOINT, FACD_SHA, LUMA, LUMA_INTEGER, STRENGTH,
    SpatialEngine, background, checked_json, chroma, digest, facd, finish_spatial, json_to, render, save,
)


def grade(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {'identity': True, 'max_luma_error': 0.0}
    if reference.shape[1] < 2:
        return baseline.copy(), {'narrow': True, 'max_luma_error': 0.0}
    halves = np.array_split(reference, 2, axis=1)
    views = [finish_spatial(target, engine.prepare(target, part).apply(target), STRENGTH) for part in halves]
    a, b = [chroma(view, view @ LUMA) for view in views]
    x = (np.arange(target.shape[1]) + .5) / target.shape[1]
    ramp = np.clip((x - .45) / .10, 0, 1)[None, :, None]
    ramp = ramp * ramp * (3 - 2 * ramp)
    output, compressed = render((1 - ramp) * a + ramp * b, baseline @ LUMA)
    return output, dict(max_luma_error=float(np.abs((output.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000),
                        half_chroma_mae=float(np.abs(a - b).mean()), gamut_fraction=compressed)


def layout(engine, out):
    parent = BASE / 'reference-layout-collision-v1'
    source = BASE / 'mls-linear-pilot'
    edited = BASE / 'mls-film-light-cross-v1'
    old = checked_json(parent / 'results.json', LAYOUT_SHA)
    original = checked_json(source / 'results.json', SOURCE_SHA)
    authored = checked_json(edited / 'results.json', PARENT_SHA)
    out.mkdir()
    results = {}
    for scene, case in old['cases'].items():
        target_file = f'{scene}_2HAL_DESK.png'
        target = verified(source / target_file, original['scenes'][scene]['inputs_sha256'][target_file])
        full = verified(edited / f'{scene}-standard-reference-film.png', authored['references'][scene]['standard'])
        empty = np.zeros(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(img, empty)['neutral_proxy'] for img in (target, full)))
        if neutral.sum() != authored['cases'][scene]['neutral_pixels']:
            raise ValueError('Changed neutral coverage')
        refs, expected, result, hashes = {}, {}, {}, {}
        for side, receipt in case['image_sha256'].items():
            refs[side] = verified(parent / f'{scene}-{side}-reference.png', receipt['reference'])
            expected[side] = verified(parent / f'{scene}-{side}-expected.png', receipt['expected'])
            base = verified(parent / f'{scene}-{side}-13.png', receipt['13'])
            result[side], checks = grade(engine, target, refs[side], base)
            hashes[side] = dict(checks, sha=save(out / f'{scene}-{side}.png', result[side]))
        scores = {}
        for name, zone in (('whole', ~empty), ('neutral', neutral)):
            row = metrics(expected['left'], expected['right'], result['left'], result['right'], zone)
            row['joint'] = row['passed'] and max(row['off_matches_original_mae'], row['on_matches_authored_mae']) <= 3
            scores[name] = row
        results[scene] = dict(scores=scores, frames=hashes, board=layout_panel(
            out / f'{scene}-comparison.png', refs, target, expected, result, output_label='Local reference halves'))
    return dict(cases=results, passed=all(row['joint'] for case in results.values() for row in case['scores'].values()))


def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed checkpoint')
    mini = np.array([[[100, 50, 150], [70, 30, 20]]], dtype=np.uint8)
    np.testing.assert_array_equal(grade(engine, mini, mini, 255 - mini)[0], mini)
    out.mkdir()
    local = layout(engine, out / 'layout')
    bg = background(engine, out / 'bg-real', grade=grade, label='Reference halves')
    author = facd(engine, out / 'facd', grade=grade, label='Reference halves')
    rows = [row for suite in (bg, author) for case in suite['cases'].values() for row in case['outputs'].values()]
    rows.extend(row for case in local['cases'].values() for row in case['frames'].values())
    max_error = max(row['max_luma_error'] for row in rows)
    json_to(out / 'results.json', dict(checkpoint=CHECKPOINT, sources=[LAYOUT_SHA, BG_SHA, FACD_SHA],
            script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name(
                'reference-half-layout-protocol.md')), layout=local, background=bg, facd=author,
            max_luma_error=max_error, passed=local['passed'] and bg['passed'] and author['passed'] and max_error <= .5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
