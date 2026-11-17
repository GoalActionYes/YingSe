"""Isolate spatial switching by rendering each reference color table on the whole target."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_psbench_action_rgb_continuity import adjacency, continuous_delta
from probe_psbench_action_rgb_interpolation import make_grid
from probe_psbench_action_rgb_transfer import color_keys, score
from probe_psbench_teal_portrait import verify_source
from reference_color_app.media import decode_image


def checked_image(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Comparison image changed: {path.name}')
    with Image.open(path) as image:
        return np.array(image)


def run(folder, out, pair_evidence, hard_evidence, continuous_evidence):
    pair, hard, continuous = [
        json.loads(path.read_text(encoding='utf-8'))
        for path in (pair_evidence, hard_evidence, continuous_evidence)
    ]
    if verify_source(folder, '105') != pair['files'] or pair['strength'] != .8:
        raise ValueError('Unexpected source files or blend strength')
    if digest(pair_evidence) != continuous['pair_sha256'] or digest(hard_evidence) != continuous['hard_sha256']:
        raise ValueError('Baseline evidence does not match the prior continuity run')
    source, edited = (decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end'))
    if source.shape != edited.shape or source.shape != (1080, 1440, 3):
        raise ValueError('Unexpected collage geometry')
    rows, cols = np.ogrid[:1080, :704]
    folds = ((rows // 32 + cols // 32) % 2).astype(bool)
    out.mkdir(exist_ok=False)
    cases, artifacts = {}, {}
    for target, reference in (('portrait', 'tower'), ('tower', 'portrait')):
        left, right = pair['crop_columns'][target]
        ref_left, ref_right = pair['crop_columns'][reference]
        original = source[:, left:right]
        ref = source[:, ref_left:ref_right]
        ref_delta = np.rint((edited[:, ref_left:ref_right].astype(np.int16) - ref) * .8).astype(np.int16)
        variants = {'original': original, 'current13': checked_image(
            folder / 'action-pair-v1' / f'{target}.png', pair['predictions'][target]
        )}
        switched_path = folder / 'action-continuity-v1' / f'{target}-continuous.png'
        variants['switched'] = checked_image(switched_path, continuous['artifacts'][switched_path.name])
        reference_keys = color_keys(ref)
        for index, train in enumerate((~folds, folds)):
            role = f'fixed_{index}'
            grid = make_grid(reference_keys[train], ref_delta[train])
            change = continuous_delta(original.reshape(-1, 3), *grid).reshape(original.shape)
            rendered = np.rint(original + change).clip(0, 255).astype(np.uint8)
            variants[role] = rendered
            path = out / f'{target}-{role}.png'
            Image.fromarray(rendered).save(path)
            artifacts[path.name] = digest(path)
        switched = np.where(folds[..., None], variants['fixed_0'], variants['fixed_1'])
        if not np.array_equal(switched, variants['switched']):
            raise ValueError(f'{target}: prior spatial switching did not reproduce')

        support, added = (
            checked_image(
                folder / 'action-interpolation-v1' / f'{target}-{role}.png', hard['artifacts'][f'{target}-{role}.png']
            ) == 255 for role in ('supported', 'added')
        )
        delta = np.rint((edited[:, left:right].astype(np.int16) - original) * .8).astype(np.int16)
        expected = original + delta
        scores = {label: score(original, expected, variants, mask) for label, mask in
                  [('all', np.ones_like(support)), ('added', added)]}
        edges = adjacency(original, expected, {role: variants[role] for role in ('switched', 'fixed_0', 'fixed_1')},
                          support)
        boundary = edges['changed_support']['p95']
        gates = {}
        for method in ('fixed_0', 'fixed_1'):
            quality = True
            for scope, region in [('all', 'whole'), ('all', 'neutral_proxy'), ('added', 'neutral_proxy')]:
                methods = scores[scope][region]['methods']
                result = methods[method]
                quality &= result['mae'] <= min(methods['original']['mae'], methods['current13']['mae']) - 1
                quality &= (result['projection'] or 0.) > 0
                if scope == 'all':
                    quality &= result['mae'] <= methods['switched']['mae'] + .5
            gates[method] = dict(quality=quality, continuity=boundary[method] <= boundary['author'] + 2)
        cases[target] = dict(scores=scores, adjacency=edges, gates=gates, prior_pixels_reproduced=folds.size)
    report = dict(cases=cases, artifacts=artifacts,
                  numeric_gates_passed=all(all(gate.values()) for case in cases.values()
                                           for gate in case['gates'].values()),
                  upstream_sha256={path.name: digest(path) for path in
                                   (pair_evidence, hard_evidence, continuous_evidence)},
                  script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-action-rgb-global-map-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: dict(edges=case['adjacency']['changed_support'], gates=case['gates'])
                      for name, case in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    for name in ('pair_evidence', 'hard_evidence', 'continuous_evidence'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
