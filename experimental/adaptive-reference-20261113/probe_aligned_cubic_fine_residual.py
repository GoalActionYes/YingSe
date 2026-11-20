"""Check repeated-capture rejection using fine residuals in matched flat blocks."""
import argparse
from io import BytesIO
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_aligned_cubic_capture_stress import Stress, exposure_pairs, paletted_cases
from probe_aligned_cubic_pilot import certify
from probe_arri_shared_look import metrics
from probe_frozen_render_blend_bound import neutral
from probe_playraw_mixed_light_transfer import LUMA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial


ARCHIVE_SHA = 'b0ab4559c6194e6c204d94582c332bcc709c3fbf4e294fbca28346c051d53b52'


def blocks(rgb):
    h, w = rgb.shape[:2]
    y = rgb[: h - h % 4, : w - w % 4] @ LUMA
    cells = y.reshape(h // 4, 2, 2, w // 4, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    average = cells.mean(axis=(-1, -2)).reshape(h // 4, w // 4, 4)
    usable = ((average.min(axis=-1) >= 12) & (average.max(axis=-1) <= 243)
              & (np.ptp(average, axis=-1) <= 3))
    residual = cells[..., 0, 0] + cells[..., 1, 1] - cells[..., 0, 1] - cells[..., 1, 0]
    return usable, residual


def fine_residual(prediction, reference):
    a, x = blocks(prediction)
    b, y = blocks(reference)
    common = a & b
    count = int(common.sum())
    if count < 64:
        return dict(blocks=count, reason='few_blocks', passed=False)
    x, y = x[common].ravel(), y[common].ravel()
    deviations = [float(x.std()), float(y.std())]
    if min(deviations) < .25:
        return dict(blocks=count, deviations=deviations, reason='weak_residual', passed=False)
    correlation = float(np.corrcoef(x, y)[0, 1])
    passed = correlation >= .65
    return dict(blocks=count, deviations=deviations, correlation=correlation,
                reason='matched' if passed else 'low_correlation', passed=passed)


def recode(rgb):
    with BytesIO() as stream:
        Image.fromarray(rgb).save(stream, format='JPEG', quality=85, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as encoded:
            return np.array(encoded)


def jpeg_diagnostic(target, reference):
    target, reference = recode(target), recode(reference)
    previous, table = certify(target, reference)
    result = None if table is None else fine_residual(apply_mapping(target, table, 1), reference)
    return dict(route=previous['route'], residual=result)


def fixtures():
    for name, before, after in exposure_pairs():
        yield 'negatives', name, before, after
    anchors = {}
    for family, scene, states, original in paletted_cases():
        anchor = states[original][0]
        anchors.setdefault(family, {})[scene] = anchor
        for state, (raw, graded) in states.items():
            name = f'{family}-{scene}-{state}'
            yield 'positives', name, raw, graded
            if state != original:
                yield 'negatives', f'{name}-to-anchor', raw, anchor
                yield 'negatives', f'{name}-from-anchor', anchor, raw
    for family, group in anchors.items():
        for first, second in permutations(group, 2):
            yield 'negatives', f'{family}-{first}-from-{second}', group[first], group[second]


@torch.inference_mode()
def run(out):
    archive = BASE / 'aligned-cubic-capture-stress-v1'
    before = checked_json(archive / 'results.json', ARCHIVE_SHA)
    out.mkdir()
    torch.set_num_threads(4)
    base = Stress(out)
    rows = {'negatives': {}, 'positives': {}}
    for group, name, target, reference in fixtures():
        existing, table = certify(target, reference)
        prior = before[group][name]
        if existing['route'] != prior['decision']['route']:
            raise ValueError(f'Frozen route changed: {name}')
        positive = group == 'positives'
        if positive and certify(target, target)[0]['route'] != 'identity':
            raise ValueError('Exact identity route changed')
        row = {'route_before': existing['route'], 'route_after': existing['route']}
        rows[group][name] = row
        if table is None:
            continue
        proposed = apply_mapping(target, table, 1)
        cubic = finish_spatial(target, proposed, STRENGTH)
        suffix = 'positive' if positive else 'cubic'
        expected_hash = prior['output_sha256'] if positive else prior['output_sha256']['cubic']
        np.testing.assert_array_equal(cubic, checked_rgb(archive / f'{name}-{suffix}.png', expected_hash))
        evidence = fine_residual(proposed, reference)
        row.update(residual=evidence, jpeg=jpeg_diagnostic(target, reference))
        final = cubic
        if not evidence['passed']:
            row['route_after'] = 'legacy13'
            final = base.baseline(target, reference)
        row['sha256'] = save(out / f'{name}.png', final)
        if positive:
            expected = finish_spatial(target, reference, STRENGTH)
            selected = neutral(target) & neutral(reference)
            scores = {'whole': metrics(target, expected, target, final, np.ones(selected.shape, dtype=bool))}
            scores['neutral'] = metrics(target, expected, target, final, selected) if selected.sum() >= 256 else None
            row['scores'] = scores
        else:
            normal = checked_rgb(archive / f'{name}-13.png', prior['output_sha256']['13'])
            row['matches_normal'] = bool(np.array_equal(final, normal))
            row['comparison_sha256'] = panel(out / f'{name}-comparison.png', {
                'physical recapture': {'target': target, 'reference': reference, '13': normal,
                                       'old cubic': cubic, 'gated': final}})
    if {key: len(group) for key, group in rows.items()} != {'negatives': 47, 'positives': 14}:
        raise ValueError('Fixture count changed')
    activated_negatives = [row for row in rows['negatives'].values() if row['route_before'] == 'cubic']
    retained = [row for row in rows['positives'].values() if row['route_after'] == 'cubic']
    passed = (len(activated_negatives) == 2 and all(row['matches_normal'] for row in activated_negatives)
              and len(retained) >= 6 and all(score['passed'] for row in retained
                                           for score in row['scores'].values() if score is not None))
    json_to(out / 'results.json', dict(archive_sha256=ARCHIVE_SHA, script_sha256=digest(Path(__file__)),
                                      positive_cubic_retained=len(retained), passed=passed, **rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
