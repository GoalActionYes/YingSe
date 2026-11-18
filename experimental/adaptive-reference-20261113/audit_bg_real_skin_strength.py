"""Enumerate every RGB8 result between the frozen base and protection endpoints."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import time

import numpy as np

from audit_bg_real_skin_regions import BASE, PROTOCOL, ROOT, digest, read


COEFFICIENTS = np.array([299, 587, 114], dtype=np.int64)


def enumerate_weights(delta):
    edges = {Fraction(0), Fraction(1)}
    for change in np.unique(np.abs(delta)):
        if change:
            edges.update(Fraction(k, 2 * int(change)) for k in range(1, 2 * int(change), 2))
    ordered = sorted(edges)
    centers = [(a+b)/2 for a, b in zip(ordered, ordered[1:])]
    return sorted(edges.union(centers)), len(ordered)


def mix(first, delta, weights):
    numerators = np.array([w.numerator for w in weights], dtype=np.int64)[:, None, None]
    denominators = np.array([w.denominator for w in weights], dtype=np.int64)[:, None, None]
    lo, remainder = np.divmod(first[None] * denominators + delta[None] * numerators, denominators)
    double = remainder * 2
    return lo + ((double > denominators) | ((double == denominators) & (lo % 2 == 1)))


def chroma_codes(values):
    return 1000 * values - (values @ COEFFICIENTS)[..., None]


def screen(start, end, expected, spans):
    delta = end-start
    weights, breakpoints = enumerate_weights(delta)
    c1, f1, base, whole = (mix(first, diff, weights) for first, diff in zip(start, delta, strict=True))
    scores = np.abs(chroma_codes(f1-c1))
    direction = chroma_codes(expected)
    projected = (chroma_codes(whole-base) * direction).sum(axis=(1, 2))
    direction_power = int((direction ** 2).sum())
    valid_style = (projected > 0) & (projected * 10 >= projected[0] * 9)
    region_scores = np.stack([scores[:, left:right].sum(axis=(1, 2)) for left, right in spans], axis=1)
    pixel_counts = np.array([right-left for left, right in spans])
    valid = valid_style & np.all(region_scores <= region_scores[0]+750*pixel_counts, axis=1)
    hits = (100 * region_scores <= 85 * region_scores[0]).sum(axis=1)
    assert weights[0] == 0 and weights[-1] == 1 and valid[0]
    best = int(hits[valid].max())
    max_allowed = int(np.nonzero(valid)[0][-1])
    witnesses = np.nonzero(valid & (hits == best))[0]
    best_index = int(witnesses[np.argmin(region_scores[witnesses].sum(axis=1))])
    return dict(breakpoints=breakpoints, states_tested=len(weights), safe_style_states=int(valid.sum()),
                maximum_passing_patches=best, maximum_feasible_weight=str(weights[max_allowed]),
                best_weight=str(weights[best_index]),
                baseline_direction_projection=float(projected[0]/direction_power),
                direction_projection_at_best=float(projected[best_index]/direction_power),
                fully_protected_direction_projection=float(projected[-1]/direction_power),
                baseline_chroma_mae=(region_scores[0]/(3000*pixel_counts)).tolist(),
                best_choice_chroma_mae=(region_scores[best_index]/(3000*pixel_counts)).tolist())


def run(out):
    start = time.monotonic()
    docs = ROOT/'docs/research'
    names = ('bg-real-pixel-pilot', 'bg-real-legacy-skin', 'local-edit-toggle')
    files = [docs/f'2026-11-18-{name}-evidence.json' for name in names]
    original, protected, toggle = [json.loads(path.read_bytes()) for path in files]
    assert protected['input_evidence_sha256'] == digest(files[0])
    assert protected['toggle_sha256'] == digest(files[2])
    assert protected['skin_regions_protocol_sha256'] == digest(PROTOCOL)
    patches = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                         PROTOCOL.read_text(encoding='utf-8'), re.MULTILINE)
    records = {}
    for key, row in original['cases'].items():
        pair = row['target_pair']
        boxes, names = [], []
        for target, reference, name, *bounds in patches:
            if f'{target}-{reference}' != pair or not protected['metrics'][pair][name]['direct']:
                continue
            left, top, right, bottom = map(int, bounds)
            boxes.append(np.s_[top:bottom, left:right])
            names.append(name)
        if not boxes:
            continue

        def pixels(path, expected):
            image = read(path, expected)
            return np.concatenate([image[box].reshape(-1, 3) for box in boxes]).astype(np.int64)

        folder = BASE/'local-edit-toggle-v1'/pair
        expected = toggle['cases'][pair]['files']
        candidate_hashes = protected['output_sha256']
        photo_items = [f'{key}-{role}.png' for role in ('C1', 'F1')]
        local_items = [f'{pair}-{role}.png' for role in ('base', 'whole')]
        first = [pixels(BASE/'bg-real-pixel-pilot'/name, row['outputs'][role])
                 for name, role in zip(photo_items, ('C1', 'F1'), strict=True)]
        first.extend(pixels(folder/name, expected[name]) for name in ('base-current13.png', 'whole-current13.png'))
        second = [pixels(BASE/'bg-real-legacy-skin'/name, candidate_hashes[name])
                  for name in photo_items + local_items]
        intended = [pixels(folder/name, expected[name]) for name in ('base-expected.png', 'whole-expected.png')]
        lengths = np.cumsum([0, *((box[0].stop-box[0].start)*(box[1].stop-box[1].start) for box in boxes)])
        spans = list(zip(lengths[:-1], lengths[1:], strict=True))
        record = screen(np.stack(first), np.stack(second), intended[1]-intended[0], spans)
        records[pair] = dict(patches=names, **record)
    assert len(records) == 3 and sum(len(r['patches']) for r in records.values()) == 5
    if time.monotonic() - start > 30:
        raise TimeoutError('CPU budget exceeded')
    count = sum(row['maximum_passing_patches'] for row in records.values())
    source = Path(__file__)
    result = dict(records=records, maximum_passing_patches=count, proceed=count >= 4,
                  parents={path.name: digest(path) for path in files},
                  source_sha256=digest(source), protocol_sha256=digest(source.with_name(
                      'bg-real-skin-strength-bound-protocol.md')))
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(count, result['proceed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
