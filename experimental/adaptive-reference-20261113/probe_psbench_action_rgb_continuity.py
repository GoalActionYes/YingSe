"""Replay the PSBench color nodes with a continuous identity fallback."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_psbench_action_rgb_interpolation import interpolate, make_grid
from probe_psbench_action_rgb_transfer import color_keys, score
from probe_psbench_teal_portrait import verify_source
from reference_color_app.media import decode_image


def continuous_delta(rgb, grid, known):
    # Missing edit values are zero, so no division or threshold changes the blend.
    return interpolate(rgb, np.where(known[..., None], grid, 0.), np.ones_like(known))[0]


def adjacency(original, expected, variants, support):
    original = original.astype(np.int16)
    edits = [rgb.astype(np.int16) - original for rgb in (expected, *variants.values())]
    names = ['author', *variants]
    groups = {key: [] for key in ('all', 'changed_support', 'both_supported')}
    for front, back in ((np.s_[1:], np.s_[:-1]), (np.s_[:, 1:], np.s_[:, :-1])):
        smooth = np.max(np.abs(original[front] - original[back]), axis=-1) <= 3
        sizes = np.stack([np.max(np.abs(edit[front] - edit[back]), axis=-1) for edit in edits], axis=-1)
        for name, mask in [('all', smooth),
                           ('changed_support', smooth & (support[front] ^ support[back])),
                           ('both_supported', smooth & support[front] & support[back])]:
            groups[name].append(sizes[mask])
    result = {}
    for name, values in groups.items():
        merged = np.concatenate(values)
        if not len(merged):
            raise ValueError(f'No adjacent pixels for {name}')
        result[name] = dict(pairs=len(merged), p95=dict(zip(names, np.percentile(merged, 95, axis=0).tolist())))
    return result


def run(folder, out, pair_evidence, hard_evidence):
    pair = json.loads(pair_evidence.read_text(encoding='utf-8'))
    hard = json.loads(hard_evidence.read_text(encoding='utf-8'))
    if verify_source(folder, '105') != pair['files'] or pair['strength'] != .8:
        raise ValueError('Unexpected source files or blend strength')
    if digest(pair_evidence) != hard['baseline_sha256']:
        raise ValueError('The fixed hard-cut output came from another baseline')
    source, edited = (decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end'))
    if source.shape != edited.shape or source.shape != (1080, 1440, 3):
        raise ValueError('Unexpected collage geometry')

    scenes = {}
    for name, (left, right) in pair['crop_columns'].items():
        image = source[:, left:right]
        delta = np.rint((edited[:, left:right].astype(np.int16) - image) * .8).astype(np.int16)
        base = folder / 'action-pair-v1' / f'{name}.png'
        if digest(base) != pair['predictions'][name]:
            raise ValueError(f'{name}: current13 baseline changed')
        old_files = {}
        for role in ('cross_scene', 'supported', 'added'):
            path = folder / 'action-interpolation-v1' / f'{name}-{role}.png'
            if digest(path) != hard['artifacts'][path.name]:
                raise ValueError(f'{name}: fixed {role} changed')
            with Image.open(path) as png:
                old_files[role] = np.array(png)
        scenes[name] = dict(image=image, delta=delta, keys=color_keys(image),
                            variants=dict(original=image, current13=decode_image(base),
                                          hard_cut=old_files['cross_scene'],
                                          continuous=image.copy(), same_scene=image.copy()),
                            support=old_files['supported'] == 255, added=old_files['added'] == 255)

    rows, cols = np.ogrid[:1080, :704]
    folds = ((rows // 32 + cols // 32) % 2).astype(bool)
    for test in (folds, ~folds):
        grids = {name: make_grid(scene['keys'][~test], scene['delta'][~test]) for name, scene in scenes.items()}
        for name, reference in (('portrait', 'tower'), ('tower', 'portrait')):
            scene = scenes[name]
            rgb = scene['image'][test]
            for method, grid in [('continuous', grids[reference]), ('same_scene', grids[name])]:
                scene['variants'][method][test] = np.rint(rgb + continuous_delta(rgb, *grid)).clip(0, 255)

    out.mkdir(exist_ok=False)
    cases, artifacts = {}, {}
    for name, scene in scenes.items():
        rgb = scene['image']
        expected = rgb + scene['delta']
        variants = scene['variants']
        for role in ('continuous', 'same_scene'):
            path = out / f'{name}-{role}.png'
            Image.fromarray(variants[role]).save(path)
            artifacts[path.name] = digest(path)
        scores = {label: score(rgb, expected, variants, mask) for label, mask in
                  [('all', np.ones_like(scene['support'])), ('added', scene['added'])]}
        edges = adjacency(rgb, expected, {role: variants[role] for role in ('hard_cut', 'continuous')},
                          scene['support'])
        boundary = edges['changed_support']['p95']
        quality_gate = True
        for scope, region in [('all', 'whole'), ('all', 'neutral_proxy'), ('added', 'neutral_proxy')]:
            methods = scores[scope][region]['methods']
            result = methods['continuous']
            quality_gate &= result['mae'] <= min(methods['original']['mae'], methods['current13']['mae']) - 1
            quality_gate &= (result['projection'] or 0.) > 0
            if scope == 'all':
                quality_gate &= result['mae'] <= methods['hard_cut']['mae'] + .5
        cases[name] = dict(scores=scores, adjacency=edges, quality_gate=quality_gate,
                           continuity_gate=boundary['continuous'] <= boundary['author'] + 2)
    report = dict(cases=cases, artifacts=artifacts,
                  numeric_gates_passed=all(case['quality_gate'] and case['continuity_gate']
                                           for case in cases.values()),
                  pair_sha256=digest(pair_evidence), hard_sha256=digest(hard_evidence),
                  script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-action-rgb-continuity-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: case['adjacency']['changed_support'] for name, case in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('pair_evidence', type=Path)
    parser.add_argument('hard_evidence', type=Path)
    run(**vars(parser.parse_args()))
