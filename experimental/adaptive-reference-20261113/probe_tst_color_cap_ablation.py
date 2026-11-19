"""Locate color response loss before or after the frozen per-pixel cap."""

import argparse
from pathlib import Path

import numpy as np

from probe_tst_donor_supervision import (
    BASE, LUMA, RENDER_SHA, SPLIT, SPLIT_SHA, checked_rgb, features, get_color, images,
    json_to, load, masks, measure, neutral, render, save, tensor_samples, whole,
)


PARENT = BASE.parent / 'tst-scale-agreement-supervision-v1'
PARENT_SHA = 'a2b697386a1d4e942fcd4f31ba9ef25d5e59922645a686cb5f65dc8c4cc3cb85'


def stages(target, reference, frame, coef, area):
    base = get_color(frame)
    if np.array_equal(target, reference):
        color = get_color(target)
        return target.copy(), target.copy(), color, color, np.zeros_like(area)
    original = get_color(target)[area]
    candidate = features(target, reference, frame, area) @ coef
    length = np.linalg.norm(candidate, axis=1)
    limit = 2 * np.linalg.norm(base[area] - original, axis=1)
    factor = np.divide(limit, length, out=np.ones_like(length), where=length > 0)
    light = (frame @ LUMA)[area]
    hit = np.zeros_like(area)
    hit[area] = length > limit
    colors, frames = [], []
    bounded = candidate * np.minimum(factor, 1)[:, None]
    for vector in (candidate, bounded):
        color, output = base.copy(), frame.copy()
        color[area] = original + vector
        output[area], _ = render(color[area], light)
        colors.append(color)
        frames.append(output)
    return *frames, *colors, hit


def contrast(colors, truth, area):
    a, b, c = colors
    expected = (get_color(truth[2]) - get_color(truth[0]))[area]
    actual = (c - .5 * (a + b))[area]
    want = float(np.linalg.norm(expected))
    value = float(np.linalg.norm(actual))
    dot = float(np.sum(actual * expected))
    return {'projection': dot / want**2, 'cosine': dot / (value * want) if value else None,
            'response_norm_ratio': value / want}


def run(out):
    old = load(PARENT / 'results.json', PARENT_SHA)
    split = load(SPLIT / 'private-assignment.json', SPLIT_SHA)
    archive = BASE / 'reference-equivalence-development-baseline-v1'
    frozen = load(archive / 'results.json', RENDER_SHA)
    coef = np.array(old['coefficients'])[:, -2:]
    cases = {}
    out.mkdir()
    for index, tensors in enumerate(tensor_samples(split['development'])):
        target, references, truth = images(tensors)
        area = neutral(target)
        parent = old['cases'][str(index)]['agreement']
        base = [checked_rgb(archive / f'{index}-{i}.png', frozen['predictions'][f'{index}-{i}.png'])
                for i in range(3)]
        items = [stages(target, ref, frame, coef, area)
                 for ref, frame in zip(references, base, strict=True)]
        uncapped, capped, raw_colors, capped_colors, hit = zip(*items, strict=True)
        for slot, (frame, digest) in enumerate(zip(capped, parent['hashes'], strict=True)):
            stored = checked_rgb(PARENT / f'{index}-{slot}-agreement.png', digest)
            np.testing.assert_array_equal(frame, stored)
        case = {'whole': whole(uncapped, base, truth), 'hashes': [
            save(out / f'{index}-{slot}.png', frame) for slot, frame in enumerate(uncapped)]}
        if 'local' in parent:
            regions = masks(target, (truth[0], truth[2]))
            active = regions[0] | regions[1]
            case['local'] = {
                'baseline': measure(base, (truth[0], truth[2]), regions),
                'capped': parent['local'],
                'uncapped': measure(uncapped, (truth[0], truth[2]), regions),
                'raw_color': contrast(raw_colors, truth, active),
                'capped_color': contrast(capped_colors, truth, active),
                'cap_share': [float(mask[active].mean()) for mask in hit],
            }
        cases[str(index)] = case
    local = [row['local']['uncapped']['passed'] for row in cases.values() if 'local' in row]
    if len(local) != 5 or len(cases) != 24:
        raise ValueError('Frozen development set changed')
    summary = {'local_passed': sum(local), 'whole_passed': sum(r['whole']['passed'] for r in cases.values()),
               'worst_extra_mae': max(r['whole']['worst_regression'] for r in cases.values())}
    passed = summary['local_passed'] >= 3 and summary['whole_passed'] >= 7 and summary['worst_extra_mae'] <= 2
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'cases': cases, 'summary': summary, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
