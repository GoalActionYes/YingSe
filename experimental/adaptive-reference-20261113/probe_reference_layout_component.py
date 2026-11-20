"""Separate remaining tone and chroma error in the frozen layout experiment."""
import argparse
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_appearance_layout_response import LAYOUT_SHA, PARENT_SHA, SOURCE_SHA, verified
from probe_arri_shared_look import metrics
from probe_reference_layout_collision import panel as board
from probe_target_mirror_chroma import BASE, LUMA, checked_json, chroma, digest, json_to, render, save

PARENT = 'c336e00f1c73bd096619b5a2d98f9ae04ae3c0873dfb0d39270953427d1ca78a'
SIDES = ('left', 'right')


def score(first, second, expected, regions):
    rows = {}
    for name, region in regions.items():
        row = metrics(expected['left'], expected['right'], first, second, region)
        row['joint'] = row['passed'] and max(row['off_matches_original_mae'], row['on_matches_authored_mae']) <= 3
        rows[name] = row
    return rows


def errors(actual, ideal, zones):
    delta = actual.astype(float) - ideal
    light = delta @ LUMA
    color = delta - light[..., None]
    np.testing.assert_allclose(light[..., None] + color, delta, rtol=0, atol=1e-12)
    return {name: dict(luma_mae=float(np.abs(light[region]).mean()),
                       chroma_rgb_mae=float(np.abs(color[region]).mean())) for name, region in zones.items()}


def run(out):
    earlier = BASE / 'reference-half-layout-v1'
    archived = checked_json(earlier / 'results.json', PARENT)
    staged = BASE / 'reference-layout-collision-v1'
    layout = checked_json(staged / 'results.json', LAYOUT_SHA)
    source = BASE / 'mls-linear-pilot'
    images = checked_json(source / 'results.json', SOURCE_SHA)
    edits = BASE / 'mls-film-light-cross-v1'
    authored = checked_json(edits / 'results.json', PARENT_SHA)
    out.mkdir()
    results = {}
    for scene, case in layout['cases'].items():
        file = f'{scene}_2HAL_DESK.png'
        target = verified(source / file, images['scenes'][scene]['inputs_sha256'][file])
        full = verified(edits / f'{scene}-standard-reference-film.png', authored['references'][scene]['standard'])
        empty = np.zeros(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(img, empty)['neutral_proxy'] for img in (target, full)))
        if neutral.sum() != authored['cases'][scene]['neutral_pixels']:
            raise ValueError('Changed neutral region')
        zones = dict(whole=~empty, neutral=neutral)
        expected, refs, original = {}, {}, {}
        choices = {'ideal_luma': {}, 'ideal_chroma': {}}
        diagnostics = {}
        for side in SIDES:
            hashes = case['image_sha256'][side]
            refs[side] = verified(staged / f'{scene}-{side}-reference.png', hashes['reference'])
            ideal = verified(staged / f'{scene}-{side}-expected.png', hashes['expected'])
            actual = verified(earlier / 'layout' / f'{scene}-{side}.png',
                              archived['layout']['cases'][scene]['frames'][side]['sha'])
            expected[side], original[side] = ideal, actual
            actual_y, ideal_y = actual @ LUMA, ideal @ LUMA
            diagnostics[side] = dict(components=errors(actual, ideal, zones), candidates={})
            for name, (color, light) in {'ideal_luma': (chroma(actual, actual_y), ideal_y),
                                         'ideal_chroma': (chroma(ideal, ideal_y), actual_y)}.items():
                result, compressed = render(color, light)
                choices[name][side] = result
                diagnostics[side]['candidates'][name] = dict(
                    compressed=compressed, sha=save(out / f'{scene}-{side}-{name}.png', result))
        scores = {name: score(*frames.values(), expected, zones) for name, frames in choices.items()}
        baseline = score(*original.values(), expected, zones)
        if baseline != archived['layout']['cases'][scene]['scores']:
            raise ValueError('Changed baseline scoring')
        results[scene] = dict(baseline=baseline, scores=scores, frames=diagnostics,
                              panels={name: board(out / f'{scene}-{name}-comparison.png',
                                                  refs, target, expected, frames, output_label=name)
                                      for name, frames in choices.items()})
    json_to(out / 'results.json', dict(source=PARENT, script_sha=digest(Path(__file__)),
            protocol_sha=digest(Path(__file__).with_name('reference-layout-component-protocol.md')), cases=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
