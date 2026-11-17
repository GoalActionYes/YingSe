"""Test a continuous image-context correction against frozen PSBench RGB tables."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter

from background_intervention import target_regions
from independent_coco import digest
from probe_psbench_action_rgb_global_map import checked_image
from probe_psbench_action_rgb_transfer import color_keys, score
from probe_psbench_teal_portrait import verify_source
from probe_reference_chroma_spread import projection
from reference_color_app.media import decode_image


def local_contrast(rgb):
    luma = rgb.astype(np.float64) @ np.array([.2126, .7152, .0722])
    grayscale = Image.fromarray(np.rint(luma).astype(np.uint8))
    surrounding = np.asarray(grayscale.filter(ImageFilter.GaussianBlur(16)), dtype=np.float64)
    return surrounding - np.asarray(grayscale, dtype=np.float64)


def center_by_color(keys, context, change):
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    signals = np.column_stack((context, change))
    totals = np.column_stack([np.bincount(groups, weights=column) for column in signals.T])
    valid = counts[groups] >= 4
    centered = signals[valid] - (totals / counts[:, None])[groups[valid]]
    return centered[:, 0], centered[:, 1:], valid


def slope(context, change):
    energy = context @ context
    if energy <= 0:
        raise ValueError('No within-color local contrast variation')
    return (context @ change) / energy


def within_color_scores(context, change, coefficients, mask):
    fits = {name: context[:, None] * fit for name, fit in coefficients.items()}
    fits['color_only'] = np.zeros_like(change)
    return dict(pixels=int(mask.sum()), methods={
        name: dict(mae=float(np.abs(prediction[mask] - change[mask]).mean()),
                   projection=projection(prediction, change, mask))
        for name, prediction in fits.items()
    })


def smooth_edge_p95(original, prediction):
    rgb = original.astype(np.int16)
    edit = prediction.astype(np.int16) - rgb
    changes = []
    for axis in (0, 1):
        smooth = np.max(np.abs(np.diff(rgb, axis=axis)), axis=-1) <= 3
        changes.append(np.max(np.abs(np.diff(edit, axis=axis)), axis=-1)[smooth])
    return float(np.percentile(np.concatenate(changes), 95))


def worst_neutral_tile(original_error, new_error, neutral):
    increase = new_error - original_error
    regions = []
    for y in range(0, neutral.shape[0], 32):
        for x in range(0, neutral.shape[1], 32):
            tile = np.s_[y:y + 32, x:x + 32]
            mask = neutral[tile]
            if mask.sum() >= 128:
                regions.append(dict(x=x, y=y, pixels=int(mask.sum()),
                                    increase=float(increase[tile][mask].mean())))
    return max(regions, key=lambda item: item['increase'])


def synthetic_check():
    keys = np.repeat([5, 25, 84], 8)
    context = np.arange(24) + np.repeat([20, -30, 10], 8)
    offsets = np.repeat([[5, 19, 21], [0, -6, -3], [11, 1, -4]], 8, axis=0)
    expected = np.array([.3, -.2, .7])
    x, y, _ = center_by_color(keys, context, offsets + context[:, None] * expected)
    np.testing.assert_allclose(slope(x, y), expected, atol=1e-12, rtol=0)
    x, y, _ = center_by_color(keys, context, offsets)
    np.testing.assert_array_equal(slope(x, y), np.zeros(3))


def run(folder, out, pair_evidence, global_evidence):
    started = time.monotonic()
    synthetic_check()
    pair, previous = [json.loads(path.read_text(encoding='utf-8')) for path in (pair_evidence, global_evidence)]
    if (verify_source(folder, '105') != pair['files'] or pair['strength'] != .8
            or previous['upstream_sha256'][pair_evidence.name] != digest(pair_evidence)):
        raise ValueError('Unexpected source identity or strength')
    source, edited = (decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end'))
    if source.shape != edited.shape or source.shape != (1080, 1440, 3):
        raise ValueError('Unexpected source geometry')
    out.mkdir(exist_ok=False)
    rows, cols = np.ogrid[:1080, :704]
    folds = ((rows // 32 + cols // 32) % 2).astype(bool)
    scenes = {}
    for name, (left, right) in pair['crop_columns'].items():
        original = source[:, left:right]
        neutral = target_regions(original, np.zeros_like(folds))['neutral_proxy']
        scenes[name] = dict(rgb=original, columns=slice(left, right), context=local_contrast(original), neutral=neutral,
                            keys=color_keys(original))

    def author_change(item):
        difference = edited[:, item['columns']].astype(np.int16) - item['rgb']
        return np.rint(difference * .8).astype(np.int16)

    artifacts, cases = {}, {}
    for index, train in enumerate((~folds, folds)):
        for target, reference in (('portrait', 'tower'), ('tower', 'portrait')):
            item = scenes[target]
            ref = scenes[reference]
            rgb, neutral = item['rgb'], item['neutral']
            ref_context, ref_change, _ = center_by_color(
                ref['keys'][train], ref['context'][train], author_change(ref)[train]
            )
            cross_fit = slope(ref_context, ref_change)
            fixed_name = f'{target}-fixed_{index}.png'
            fixed = checked_image(folder / 'action-global-map-v1' / fixed_name, previous['artifacts'][fixed_name])
            correction = (item['context'][..., None] * cross_fit).clip(-12, 12)
            raw = np.rint(fixed + correction)
            refined = raw.clip(0, 255).astype(np.uint8)
            name = f'{target}-{index}'
            path = out / f'{name}.png'
            Image.fromarray(refined).save(path)
            artifacts[path.name] = digest(path)

            current = checked_image(folder / 'action-pair-v1' / f'{target}.png', pair['predictions'][target])
            target_change = author_change(item)
            expected = rgb + target_change
            fit_context, fit_change, _ = center_by_color(item['keys'][train], item['context'][train],
                                                         target_change[train])
            same_fit = slope(fit_context, fit_change)
            test = ~train
            context, change, valid = center_by_color(item['keys'][test], item['context'][test], target_change[test])
            centered = within_color_scores(context, change, dict(same_scene=same_fit, cross_scene=cross_fit),
                                          neutral[test][valid])
            scored = score(rgb, expected, dict(current13=current, fixed=fixed, local=refined), np.ones_like(folds))
            edges = {role: smooth_edge_p95(rgb, image) for role, image in
                     [('author', expected), ('fixed', fixed), ('local', refined)]}
            errors = [np.abs(image.astype(np.int16) - expected).mean(axis=2) for image in (fixed, refined)]
            tile = worst_neutral_tile(*errors, neutral)
            whole, n = (scored[region]['methods'] for region in ('whole', 'neutral_proxy'))
            gates = dict(neutral=n['local']['mae'] <= n['fixed']['mae'] - .50,
                         whole=whole['local']['mae'] <= whole['fixed']['mae'] + .25,
                         edge=edges['local'] <= edges['fixed'] + 1, tile=tile['increase'] <= 2)
            cases[name] = dict(scores=scored, neutral_within_color=centered, edge_p95=edges, worst_neutral_tile=tile,
                               predicted_clip_fraction=float(((raw < 0) | (raw > 255)).mean()),
                               fitted={target: same_fit.tolist(), reference: cross_fit.tolist()}, gates=gates)

    elapsed = time.monotonic() - started
    if elapsed >= 90:
        raise TimeoutError('Frozen CPU screening budget exceeded')
    report = dict(cases=cases, artifacts=artifacts, proceed=all(all(row['gates'].values()) for row in cases.values()),
                  elapsed_seconds=elapsed, pair_sha256=digest(pair_evidence), global_sha256=digest(global_evidence),
                  source_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-local-contrast-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(proceed=report['proceed'], gates={k: row['gates'] for k, row in cases.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'out', 'pair_evidence', 'global_evidence'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
