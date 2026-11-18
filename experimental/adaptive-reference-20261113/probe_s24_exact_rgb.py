"""Compare rendered edits for exactly matching RGB8 across and within photos."""
import argparse
import json
from pathlib import Path

import numpy as np

from probe_s24_calibrated_pilot import decode_image, digest, finish_spatial, image, target_regions


def keys(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.uint32), -1, 0)
    return (r << 16) | (g << 8) | b


def per_half(source, grades):
    r = np.arange(source.shape[0])[:, None] // 16
    c = np.arange(source.shape[1])[None, :] // 16
    halves = (r + c) % 2 == 0
    codes = keys(source)
    results = []
    for part in (halves, ~halves):
        values, inverse, counts = np.unique(codes[part], return_inverse=True, return_counts=True)
        eligible = counts >= 8
        means = {}
        for grade, output in grades.items():
            sums = [np.bincount(inverse, weights=channel, minlength=len(values)) for channel in output[part].T]
            means[grade] = np.stack(sums, axis=-1)[eligible] / counts[eligible, None]
        results.append(dict(values=values[eligible], count=counts[eligible], means=means))
    neutral = target_regions(source, np.zeros(source.shape[:2], bool))['neutral_proxy']
    return results, dict(whole=codes.size, neutral_proxy=int(neutral.sum()))


def compare(donor, target, grade, totals):
    halves = donor + target
    shared = halves[0]['values']
    for half in halves[1:]:
        shared = np.intersect1d(shared, half['values'], assume_unique=True)
    pixels = ((shared[:, None] >> np.array([16, 8, 0], dtype=np.uint32)) & 255).astype(np.uint8)
    neutral = target_regions(pixels[:, None, :], np.zeros((len(pixels), 1), bool))['neutral_proxy'].ravel() if len(pixels) else np.zeros(0, bool)
    indices = [np.searchsorted(half['values'], shared) for half in halves]
    a, b, x, y = [half['means'][grade][index] for half, index in zip(halves, indices)]
    result = {}
    for name, region in (('whole', np.ones(len(shared), bool)), ('neutral_proxy', neutral)):
        target_count = sum(half['count'][index][region].sum() for half, index in zip(target, indices[2:]))
        row = dict(colors=int(region.sum()), target_fraction=float(target_count / totals[name]) if totals[name] else 0.)
        if region.any():
            def distance(first, second):
                return float(np.abs(first[region] - second[region]).mean())

            row.update(cross=(distance(a, y) + distance(b, x)) / 2, donor_within=distance(a, b),
                       target_within=distance(x, y))
            row['cross_excess'] = row['cross'] - max(row['donor_within'], row['target_within'])
        result[name] = row
    return result


def run(assets, previous, out):
    prior = json.loads(previous.read_bytes())
    assert digest(assets / 'assets.json') == prior['assets_sha256']
    manifest = json.loads((assets / 'assets.json').read_bytes())

    def load(scene, grade):
        return decode_image(image(assets, manifest, scene, f'srgb_images_style_{grade}'))

    summaries, totals = {}, {}
    for scene in prior['cases']:
        source = load(scene, 1)
        grades = {grade: finish_spatial(source, load(scene, grade), .8) for grade in (2, 3)}
        summaries[scene], totals[scene] = per_half(source, grades)
    comparisons = []
    for scene, row in prior['cases'].items():
        for slot, (donor, grade) in enumerate(row['references']):
            stats = compare(summaries[donor], summaries[scene], grade, totals[scene])
            whole = stats['whole']
            adequate = whole['colors'] >= 128 and whole['target_fraction'] >= .10
            comparisons.append(dict(scene=scene, donor=donor, grade=grade, slot=slot, stats=stats,
                                    result='insufficient_coverage' if not adequate else
                                    ('cross_excess' if whole['cross_excess'] >= 2 else 'below_threshold')))
    report = dict(excluded=prior['excluded'], comparisons=comparisons,
                  previous_sha256=digest(previous), source_sha256=digest(Path(__file__)))
    out.mkdir()
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    main = next(row for row in comparisons if row['scene'] == 'img_0029' and row['grade'] == 3)
    print(json.dumps(main))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('assets', 'previous', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
