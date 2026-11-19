"""Test exact RGB8 reachability as an exclusion cue for known looks."""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_invariant_look import LABELS, evaluate, natural_scenes, render_style, renderings, samples, style_transforms
from probe_target_mirror_chroma import BASE, json_to


def reachable(transform):
    volume = np.zeros((256, 256, 256), dtype=bool)
    axis = np.arange(256, dtype=np.uint8)
    for start in range(0, 256, 8):
        rgb = np.stack(np.meshgrid(axis[start:start + 8], axis, axis, indexing='ij'), axis=-1)
        cube = render_style(rgb.reshape(-1, 256, 3), transform)
        volume[tuple(np.moveaxis(cube, -1, 0))] = True
    return volume


def tolerate_rounding(exact):
    volume = exact.copy()
    for axis in range(3):
        axes = np.moveaxis(volume, axis, 0)
        updated = axes.copy()
        updated[1:] |= axes[:-1]
        updated[:-1] |= axes[1:]
        volume = np.moveaxis(updated, 0, axis)
    return volume


def observe(rgb, tables, jpeg=False):
    if jpeg:
        stream = BytesIO()
        Image.fromarray(rgb).save(stream, format='JPEG', quality=85)
        stream.seek(0)
        with Image.open(stream) as image:
            rgb = np.array(image.convert('RGB'))
    pixels = tuple(np.moveaxis(rgb, -1, 0))
    return [float(tables[label][pixels].mean()) for label in LABELS[1:]]


def decide(scores, _):
    a, b = scores
    if a >= .995 and b < .95:
        return 'film_a', scores
    if b >= .995 and a < .95:
        return 'film_b', scores
    return 'off', scores


def summarize(fold):
    rows = [row for table in fold.values() for row in table['observations'].values()]
    summary = {'true_style_below_995': 0, 'both_at_least_995': 0, 'neither_at_995': 0, 'false_trigger': 0}
    for row in rows:
        for label, record in row['decisions'].items():
            a, b = record['scores']
            summary['both_at_least_995'] += int(min(a, b) >= .995)
            summary['neither_at_995'] += int(max(a, b) < .995)
            if label == 'off':
                summary['false_trigger'] += int(record['choice'] != 'off')
            else:
                summary['true_style_below_995'] += int(record['scores'][LABELS.index(label) - 1] < .995)
    return summary


def run(out):
    transforms = style_transforms(BASE)
    scenes = natural_scenes(BASE)
    tables, support = {}, {}
    for label, transform in transforms.items():
        exact = reachable(transform)
        tables[label] = tolerate_rounding(exact)
        if not tables[label][exact].all():
            raise ValueError('Rounding support lost reachable colors')
        support[label] = dict(exact=int(exact.sum()), tolerant=int(tables[label].sum()))
    prepared = renderings(scenes, transforms)
    out.mkdir()
    arms = {}
    for condition in ('original', 'jpeg85'):
        data = samples(BASE, scenes, transforms, lambda rgb: observe(rgb, tables, jpeg=condition == 'jpeg85'))
        observed = {}
        for domain, cohort in data.items():
            folder = out / condition / domain
            folder.mkdir(parents=True)
            observed[domain] = evaluate(cohort, domain, None, prepared, folder, decide=decide)
        if sum(row['states'] for row in observed.values()) != 72:
            raise ValueError('Unexpected set of capture states')
        arms[condition] = dict(folds=observed, summary=summarize(observed),
                               passing_domains=sum(row['passed'] for row in observed.values()))
    json_to(out / 'results.json', dict(arms=arms, support=support, passed=all(x['passing_domains'] >= 3 for x in arms.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
