"""Compare a radial residual with an equal-size color-only residual on fixed FACD folds."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_facd_filter_bank import scores
from probe_facd_two_image_baseline import STYLES, read_checked
from probe_psbench_action_rgb_continuity import continuous_delta
from probe_psbench_action_rgb_interpolation import make_grid
from probe_psbench_action_rgb_transfer import color_keys
from reference_color_app.media import decode_image


METHODS = ('tone', 'radial')


def inputs(rgb):
    h, w = rgb.shape[:2]
    y, x = (2 * np.arange(h) + 1) / h - 1, (2 * np.arange(w) + 1) / w - 1
    radius = (y[:, None] ** 2 + x ** 2) / 2
    return rgb.astype(np.float64) @ np.array([.299, .587, .114]) / 255, radius


def design(luma, radius, method):
    if method == 'tone':
        basis = (np.ones_like(luma), luma, luma ** 2, luma ** 3)
    else:
        centered, quadratic = radius - 1 / 3, radius ** 2 - 7 / 45
        basis = (centered * (1 - luma), centered * luma, quadratic * (1 - luma), quadratic * luma)
    return np.stack(basis, axis=-1).reshape(-1, 4)


def render(rgb, grid):
    delta = continuous_delta(rgb.reshape(-1, 3), *grid).reshape(rgb.shape)
    return np.rint(rgb + delta).clip(0, 255).astype(np.uint8)


def fit(training, grids):
    gram = {name: np.eye(4) * .001 for name in METHODS}
    cross = {name: np.zeros((4, 3)) for name in METHODS}
    for rgb, expected, luma, radius in training:
        residual = expected - render(rgb, grids)
        y = residual.reshape(-1, 3)
        weight = 1 / (len(training) * len(y))
        for name in METHODS:
            x = design(luma, radius, name)
            gram[name] += weight * (x.T @ x)
            cross[name] += weight * (x.T @ y)
    return {name: np.linalg.solve(gram[name], cross[name]) for name in METHODS}


def run(source, parent_outputs, registration, prior_evidence, out):
    prior = json.loads(prior_evidence.read_text(encoding='utf-8'))
    registered = json.loads(registration.read_text(encoding='utf-8'))
    if prior['registration_sha256'] != digest(registration):
        raise ValueError('Unexpected registration data')
    records = prior['selections']
    data = {name: {style: read_checked(source, name, style, registered) for style in ('Origin', *STYLES)}
            for name in records}
    coordinates = {name: inputs(group['Origin']) for name, group in data.items()}
    out.mkdir(exist_ok=False)
    artifacts, coefficients = {}, {}
    for target, row in records.items():
        train = row['training_families']
        if set(train) != set(data) - {target, *row['predictions']} or len(train) != 4:
            raise ValueError(f'Frozen split changed for {target}')
        parent = data[target]['Origin']
        (out / target).mkdir()
        pixels = np.concatenate([color_keys(data[name]['Origin']).ravel() for name in train])
        coefficients[target] = {}
        x = {method: design(*coordinates[target], method) for method in METHODS}
        for style in STYLES:
            edits = {name: np.rint((data[name][style].astype(np.int16) - data[name]['Origin']) * .8).astype(np.int16)
                     for name in train}
            table = make_grid(pixels, np.concatenate([edits[name].reshape(-1, 3) for name in train]))
            baseline = render(parent, table)
            key = f'{target}/known-{style}.png'
            path = parent_outputs / key
            if digest(path) != prior['artifacts'][key] or not np.array_equal(baseline, decode_image(path)):
                raise ValueError(f'Fixed color table no longer reproduces for {target}/{style}')
            training = [(data[name]['Origin'][::2, ::2], (data[name]['Origin'] + edits[name])[::2, ::2],
                         coordinates[name][0][::2, ::2], coordinates[name][1][::2, ::2]) for name in train]
            residuals = fit(training, table)
            coefficients[target][style] = {method: matrix.tolist() for method, matrix in residuals.items()}
            for method in METHODS:
                correction = (x[method] @ residuals[method]).reshape(parent.shape).clip(-12, 12)
                output = np.rint(baseline + correction).clip(0, 255).astype(np.uint8)
                destination = out / target / f'{style}-{method}.png'
                Image.fromarray(output).save(destination)
                artifacts[destination.relative_to(out).as_posix()] = digest(destination)
    (out / 'artifacts.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')

    assessments = {}
    for target, row in records.items():
        refs = list(row['predictions'])
        origin = data[target]['Origin']
        gold = {style: origin + np.rint((data[target][style].astype(np.int16) - origin) * .8).astype(np.int16)
                for style in STYLES}
        radius = coordinates[target][1]
        regions = dict(inner=radius <= .25, outer=radius >= .5)
        case = {}
        for method in METHODS:
            images = {style: decode_image(out / target / f'{style}-{method}.png') for style in STYLES}
            known = {style: dict.fromkeys(refs, images[style]) for style in STYLES}
            automatic = {style: {ref: images[row['predictions'][ref][style]['predicted']] for ref in refs}
                         for style in STYLES}
            spatial = {style: {region: float(np.abs((images[style].astype(np.int16) - gold[style])[mask]).mean())
                               for region, mask in regions.items()} for style in STYLES}
            case[method] = dict(known=scores(origin, gold, known, refs), automatic=scores(origin, gold, automatic, refs),
                                spatial=spatial)
        assessments[target] = case

    passes = {method: {arm: sum(all(case[method][arm]['gates'].values()) for case in assessments.values())
                       for arm in ('known', 'automatic')} for method in METHODS}
    outer_gains = {style: float(np.mean([case['tone']['spatial'][style]['outer'] -
                                         case['radial']['spatial'][style]['outer']
                                         for case in assessments.values()])) for style in STYLES}
    changes = []
    for target, case in assessments.items():
        ref = next(iter(records[target]['predictions']))
        previous = prior['cases'][target]['known_style']['errors']
        radial = case['radial']['known']['errors']
        for style in STYLES:
            old_error = previous[style]['whole']['methods'][ref]['mae']
            new_error = radial[style]['whole']['methods'][ref]['mae']
            changes.append((new_error - old_error, target, style))
    worst = max(changes)
    qualifies = passes['radial']['known'] >= 6 and sum(value >= 1 for value in outer_gains.values()) >= 2 and worst[0] <= 1
    report = dict(passes=passes, outer_gains=outer_gains, worst_whole_regression=worst, qualifies=qualifies,
                  assessments=assessments, coefficients=coefficients, artifacts=artifacts,
                  prior_sha256=digest(prior_evidence), data_sha256=digest(registration),
                  script_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'facd-radial-renderer-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passes=passes, outer_gains=outer_gains, worst=worst, qualifies=qualifies)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'parent_outputs', 'registration', 'prior_evidence', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
