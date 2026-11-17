"""Screen a closed-set filter bank with whole-photo holdout and real two-image outputs."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from probe_facd_two_image_baseline import STYLES, read_checked
from probe_psbench_action_rgb_continuity import continuous_delta
from probe_psbench_action_rgb_interpolation import make_grid
from probe_psbench_action_rgb_transfer import color_keys, score
from probe_reference_chroma_spread import projection
from reference_color_app.media import decode_image


def descriptor(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(image).reshape(-1, 3).astype(np.float64)
    luma = pixels @ np.array([.299, .587, .114]) / 255
    opponent = pixels[:, [0, 2]] - pixels[:, 1:2]
    centered = opponent - opponent.mean(axis=0)
    features = []
    for center in (.125, .375, .625, .875):
        weights = np.clip(1 - np.abs(luma - center) / .25, 0., 1.)
        if weights.sum():
            weights /= weights.sum()
            mean = weights @ centered
            std = np.sqrt(weights @ ((centered - mean) ** 2))
            features.extend((*mean, *std))
        else:
            features.extend((0., 0., 0., 0.))
    return np.asarray(features)


def classifier(samples, labels):
    samples = np.stack(samples)
    center = samples.mean(axis=0)
    scale = np.maximum(samples.std(axis=0), 1)
    standardized = (samples - center) / scale
    one_hot = np.eye(len(STYLES))[labels]
    intercept = one_hot.mean(axis=0)
    coefficients = np.linalg.solve(standardized.T @ standardized + np.eye(samples.shape[1]),
                                   standardized.T @ (one_hot - intercept))
    return center, scale, coefficients, intercept


def scores(origin, gold, variants, refs):
    masks = target_regions(origin, np.zeros(origin.shape[:2], dtype=bool))
    masks = {key: masks[key] for key in ('whole', 'neutral_proxy')}
    errors = {style: score(origin, gold[style], {'original': origin, **outputs}, masks['whole'])
              for style, outputs in variants.items()}
    differences = {style: outputs[refs[0]].astype(np.int16) - outputs[refs[1]].astype(np.int16)
                   for style, outputs in variants.items()}
    swap = {style: {region: float(np.abs(delta[mask]).mean()) if mask.any() else None
                    for region, mask in masks.items()} for style, delta in differences.items()}
    separation, change = {}, {}
    for a, b in combinations(STYLES, 2):
        key = f'{a}-{b}'
        truth = gold[b] - gold[a]
        separation[key] = float(np.abs(truth).mean())
        change[key] = {ref: {region: projection(variants[b][ref].astype(np.int16) - variants[a][ref], truth, mask)
                             for region, mask in masks.items()} for ref in refs}
    limit = max(2., .5 * min(separation.values()))
    gates = dict(stable=all(row['whole'] <= limit for row in swap.values()),
                 responds=all((item['whole'] or 0.) >= .50 for row in change.values() for item in row.values()),
                 fidelity=all(row['whole']['methods'][ref]['mae'] <= row['whole']['methods']['original']['mae'] - 1
                              for row in errors.values() for ref in refs))
    return dict(swap=swap, style_projection=change, errors=errors, swap_limit=limit, gates=gates)


def run(source, baseline, registration, out):
    prior = json.loads(baseline.read_text(encoding='utf-8'))
    registered = json.loads(registration.read_text(encoding='utf-8'))
    if digest(registration) != prior['data_sha256'] or len(prior['references']) != 7:
        raise ValueError('Unexpected frozen FACD data')
    groups = prior['references']
    photos = {family: {kind: read_checked(source, family, kind, registered) for kind in ('Origin', *STYLES)}
              for family in groups}
    features = {family: {style: descriptor(images[style]) for style in STYLES}
                for family, images in photos.items()}
    out.mkdir(exist_ok=False)
    fingerprints, selections = {}, {}
    for target, refs in groups.items():
        train = [family for family in groups if family not in (target, *refs)]
        if len(train) != 4:
            raise ValueError('Unexpected holdout split')
        center, scale, weights, bias = classifier([features[family][style] for family in train for style in STYLES],
                                                  list(range(len(STYLES))) * len(train))
        target_image = photos[target]['Origin']
        folder = out / target
        folder.mkdir()
        keys = np.concatenate([color_keys(photos[family]['Origin']).ravel() for family in train])
        table_outputs = {}
        for style in STYLES:
            edits = np.concatenate([
                np.rint((photos[family][style].astype(np.int16) - photos[family]['Origin']) * .8)
                .astype(np.int16).reshape(-1, 3) for family in train
            ])
            grid = make_grid(keys, edits)
            delta = continuous_delta(target_image.reshape(-1, 3), *grid).reshape(target_image.shape)
            table_outputs[style] = np.rint(target_image + delta).clip(0, 255).astype(np.uint8)
            path = folder / f'known-{style}.png'
            Image.fromarray(table_outputs[style]).save(path)
            fingerprints[path.relative_to(out).as_posix()] = digest(path)
        choices = {}
        for ref in refs:
            choices[ref] = {}
            for style in STYLES:
                lookup = ((features[ref][style] - center) / scale) @ weights + bias
                guessed = STYLES[int(np.argmax(lookup))]
                choices[ref][style] = dict(predicted=guessed, scores=lookup.tolist())
                path = folder / f'{ref}-{style}.png'
                Image.fromarray(table_outputs[guessed]).save(path)
                fingerprints[path.relative_to(out).as_posix()] = digest(path)
        selections[target] = dict(training_families=train, predictions=choices)
    (out / 'predictions.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')
    (out / 'artifacts.json').write_text(json.dumps(fingerprints, indent=2), encoding='utf-8')

    cases, correct = {}, 0
    for target, refs in groups.items():
        origin = photos[target]['Origin']
        authored = {style: origin + np.rint((photos[target][style].astype(np.int16) - origin) * .8).astype(np.int16)
                    for style in STYLES}
        automatic = {style: {ref: decode_image(out / target / f'{ref}-{style}.png') for ref in refs}
                     for style in STYLES}
        known = {style: dict.fromkeys(refs, decode_image(out / target / f'known-{style}.png')) for style in STYLES}
        cases[target] = dict(automatic=scores(origin, authored, automatic, refs),
                             known_style=scores(origin, authored, known, refs))
        correct += sum(style == prediction['predicted'] for values in selections[target]['predictions'].values()
                       for style, prediction in values.items())
    passes = {method: sum(all(item[method]['gates'].values()) for item in cases.values())
              for method in ('automatic', 'known_style')}
    summary = dict(classified_correct=correct, classified_total=42, gates=passes,
                   proceed=passes['automatic'] >= 6 and correct >= 34,
                   cases=cases, selections=selections, artifacts=fingerprints,
                   baseline_sha256=digest(baseline), registration_sha256=digest(registration),
                   script_sha256=digest(__file__),
                   protocol_sha256=digest(Path(__file__).with_name('facd-filter-bank-protocol.md')))
    (out / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(correct=correct, **passes, proceed=summary['proceed'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'baseline', 'registration', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
