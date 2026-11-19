"""Test a frozen correction while preserving the baseline chroma projection."""

import argparse
from pathlib import Path

import numpy as np

from probe_tst_color_cap_ablation import PARENT, PARENT_SHA, stages
from probe_tst_donor_supervision import (
    BASE, LUMA, RENDER_SHA, SPLIT, SPLIT_SHA, checked_rgb, features, get_color, images,
    json_to, load, masks, measure, neutral, render, save, tensor_samples, whole,
)


def preserve_component(base, prediction):
    length = np.linalg.norm(base, axis=1)
    direction = np.divide(base, length[:, None], out=np.zeros_like(base), where=length[:, None] > 0)
    component = np.sum(prediction * direction, axis=1)
    orthogonal = prediction - component[:, None] * direction
    parallel = np.clip(component, length, 2 * length)
    budget = np.sqrt(np.maximum(4 * length**2 - parallel**2, 0))
    norm = np.linalg.norm(orthogonal, axis=1)
    amount = np.divide(budget, norm, out=np.zeros_like(norm), where=norm > 0)
    bounded = parallel[:, None] * direction + np.minimum(amount, 1)[:, None] * orthogonal
    if np.any(np.linalg.norm(bounded, axis=1) > 2 * length + 1e-10):
        raise ValueError('Changed magnitude bound')
    if np.any(np.sum(bounded * direction, axis=1) < length - 1e-10):
        raise ValueError('Lost baseline directional component')
    return bounded


def grade(target, reference, baseline, coefficients, area):
    if np.array_equal(target, reference):
        return target.copy()
    output = baseline.copy()
    original = get_color(target)[area]
    base = get_color(baseline)[area] - original
    predicted = features(target, reference, baseline, area) @ coefficients
    vector = preserve_component(base, predicted)
    output[area], _ = render(original + vector, (baseline @ LUMA)[area])
    return output


def run(out):
    old = load(PARENT / 'results.json', PARENT_SHA)
    split = load(SPLIT / 'private-assignment.json', SPLIT_SHA)
    archive = BASE / 'reference-equivalence-development-baseline-v1'
    frozen = load(archive / 'results.json', RENDER_SHA)
    coefficients = np.array(old['coefficients'])[:, -2:]
    cases = {}
    out.mkdir()
    for index, tensors in enumerate(tensor_samples(split['development'])):
        target, references, truths = images(tensors)
        area = neutral(target)
        prior = old['cases'][str(index)]['agreement']
        baseline = [checked_rgb(archive / f'{index}-{slot}.png', frozen['predictions'][f'{index}-{slot}.png'])
                    for slot in range(3)]
        frames = []
        for slot, (reference, base) in enumerate(zip(references, baseline, strict=True)):
            control = stages(target, reference, base, coefficients, area)[1]
            np.testing.assert_array_equal(control, checked_rgb(
                PARENT / f'{index}-{slot}-agreement.png', prior['hashes'][slot]))
            frame = grade(target, reference, base, coefficients, area)
            np.testing.assert_array_equal(frame[~area], base[~area])
            frames.append(frame)
        np.testing.assert_array_equal(grade(target, target, baseline[0], coefficients, area), target)
        case = {'whole': whole(frames, baseline, truths), 'hashes': [
            save(out / f'{index}-{slot}.png', frame) for slot, frame in enumerate(frames)]}
        if 'local' in prior:
            authors = truths[0], truths[2]
            region = masks(target, authors)
            case['local'] = {'13': measure(baseline, authors, region), 'previous': prior['local'],
                             'component': measure(frames, authors, region)}
        cases[str(index)] = case
    local = [r['local']['component']['passed'] for r in cases.values() if 'local' in r]
    if len(local) != 5 or len(cases) != 24:
        raise ValueError('Changed development cases')
    summary = {'local_passed': sum(local), 'whole_passed': sum(r['whole']['passed'] for r in cases.values()),
               'worst_extra_mae': max(r['whole']['worst_regression'] for r in cases.values())}
    passed = summary['local_passed'] >= 3 and summary['whole_passed'] >= 7 and summary['worst_extra_mae'] <= 2
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'cases': cases, 'summary': summary, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
