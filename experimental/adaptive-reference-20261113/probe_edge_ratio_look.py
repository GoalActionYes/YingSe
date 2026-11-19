"""Screen reference-local color gradient ratios for known-look identification."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_invariant_look import descriptor, evaluate, fit, natural_scenes, renderings, samples, style_transforms
from probe_hdm_hdr_display import json_to


LUMA = np.array([.2126, .7152, .0722])
QUANTILES = [.10, .25, .50, .75, .90]


def edges(rgb, step):
    pairs = [(rgb[:, :-step], rgb[:, step:]), (rgb[:-step], rgb[step:])]
    pieces = []
    for left, right in pairs:
        delta = right - left
        light = delta @ LUMA
        trusted = (np.abs(light) >= 6) & np.all(delta * light[..., None] > 0, axis=-1)
        trusted &= np.all((left >= 4) & (left <= 251) & (right >= 4) & (right <= 251), axis=-1)
        pieces.append(delta[trusted][:, [0, 2]] / light[trusted, None])
    ratios = np.concatenate(pieces)
    total = sum(left.shape[0] * left.shape[1] for left, _ in pairs)
    score = np.quantile(ratios, QUANTILES, axis=0).ravel() if len(ratios) >= 32 else np.zeros(10)
    return np.r_[len(ratios) / total, score]


def describe(rgb, encoding):
    image = Image.fromarray(rgb)
    if encoding == 'jpeg85':
        with BytesIO() as payload:
            image.save(payload, 'JPEG', quality=85)
            payload.seek(0)
            with Image.open(payload) as decoded:
                image = decoded.convert('RGB')
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    values = np.array(image, dtype=np.float64)
    return descriptor(values), np.concatenate([edges(values, step) for step in (1, 4)])


def select(data, enhanced):
    return {domain: {group: {view: {label: np.r_[base, edge] if enhanced else base
                                    for label, (base, edge) in styles.items()}
                             for view, styles in views.items()}
                     for group, views in groups.items()}
            for domain, groups in data.items()}


def run(root, out):
    transforms = style_transforms(root)
    scenes = natural_scenes(root)
    prepared = renderings(scenes, transforms)
    out.mkdir()
    reports = {}
    for encoding in ('original', 'jpeg85'):
        data = samples(root, scenes, transforms, lambda rgb: describe(rgb, encoding))
        arms = {}
        for arm in ('baseline', 'edges'):
            entries = select(data, arm == 'edges')
            folds = {}
            for domain, cohort in entries.items():
                folder = out / encoding / arm / domain
                folder.mkdir(parents=True)
                model = fit([batch for key, batch in entries.items() if key != domain], 1.)
                folds[domain] = evaluate(cohort, domain, model, prepared, folder)
            if sum(row['states'] for row in folds.values()) != 72:
                raise ValueError('Changed frozen cohort')
            arms[arm] = {'folds': folds, 'passing_domains': sum(row['passed'] for row in folds.values())}
        reports[encoding] = arms
    passed = all(arm['edges']['passing_domains'] >= 3
                 and arm['edges']['passing_domains'] > arm['baseline']['passing_domains'] for arm in reports.values())
    json_to(out / 'results.json', {'encodings': reports, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
