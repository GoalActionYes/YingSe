"""Compute the per-pixel tied-weight L1 bound of the frozen local grade responses."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from independent_coco import digest
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import ROLES, mask_for, save
from probe_local_grid_capacity import score
from probe_reference_chroma_spread import BASE, PARENT, ROOT
from probe_roundtrip_grade_choice import checked_json, read_rgb
from background_intervention import edit_weight, target_regions


VARIANTS = (
    ('colorfm-l-screen-v1', '2026-11-18-colorfm-l-evidence.json', '54a289b3a67f5eb00708d035c34db5abc1784d88cd94537a685eb5c0ccff7213'),
    ('colorfm-local-pooling-v1', '2026-11-18-colorfm-local-pooling-evidence.json', 'ced085486a4cb7c649229d08c00586f033b66905497aa0d9d963d4d4e32211dd'),
)


def optimal_weights(deltas, desired):
    """A convex piecewise-linear objective attains a minimum at a cell vertex."""
    vertices = deltas.astype(np.float64)
    target = desired.astype(np.float64)
    p = vertices[1] - vertices[0]
    q = vertices[2] - vertices[0]
    rhs = target - vertices[0]
    zero, one = np.zeros_like(rhs[..., 0]), np.ones_like(rhs[..., 0])
    lines = [(one, zero, zero), (zero, one, zero), (one, one, one)]
    lines += [(p[..., c], q[..., c], rhs[..., c]) for c in range(3)]
    best = np.full(rhs.shape[:-1], np.inf)
    weights = np.zeros((*best.shape, 3))
    for (a, b, c), (d, e, f) in combinations(lines, 2):
        determinant = a*e - b*d
        independent = determinant != 0
        x = np.divide(c*e-b*f, determinant, out=np.zeros_like(zero), where=independent)
        y = np.divide(a*f-c*d, determinant, out=np.zeros_like(zero), where=independent)
        valid = independent & (x >= -1e-12) & (y >= -1e-12) & (x+y <= 1+1e-12)
        x, y = np.clip(x, 0, 1), np.clip(y, 0, 1)
        y = np.minimum(y, 1-x)
        value = np.abs(p*x[..., None] + q*y[..., None] - rhs).mean(axis=-1)
        better = valid & (value < best - 1e-12)
        best[better] = value[better]
        weights[better] = np.stack((1-x-y, x, y), axis=-1)[better]
    assert np.isfinite(best).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-12, rtol=0)
    return weights, best


def run(out):
    docs = ROOT / 'docs/research'
    frozen = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    additional = [(BASE / directory, checked_json(docs / name, sha)['outputs'])
                  for directory, name, sha in VARIANTS]
    parent = checked_json(PARENT / 'results.json', frozen['parent_sha256'])
    out.mkdir()
    cases = {}
    for row in parent['cases']:
        pair = row['pair']
        people = tuple(mask_for(row, key) for key in pair.split('-'))
        spatial = tuple(edit_weight(mask, 1)[0] if mask.any() else np.zeros(mask.shape) for mask in people)
        folder = BASE / 'local-edit-toggle-v1' / pair
        files = frozen['cases'][pair]['files']
        target = read_rgb(folder / 'target.png', files['target.png'])
        authored = {role: read_rgb(folder / f'{role}-expected.png', files[f'{role}-expected.png']) for role in ROLES}
        endpoints = {}
        for role in ROLES:
            name = f'{pair}-{role}.png'
            baseline = read_rgb(folder / f'{role}-current13.png', files[f'{role}-current13.png'])
            endpoints[role] = np.stack([baseline, *(read_rgb(root / name, hashes[name]) for root, hashes in additional)])
        deltas = endpoints['local'].astype(np.int16) - endpoints['base']
        desired = authored['local'].astype(np.int16) - authored['base']
        weights, floor = optimal_weights(deltas, desired)
        output = {role: np.rint(np.einsum('khwc,hwk->hwc', frames, weights)).astype(np.uint8)
                  for role, frames in endpoints.items()}
        renders = {role: save(out / f'{pair}-{role}.png', image) for role, image in output.items()}
        neutral = target_regions(target, people[0])['neutral_proxy'] & people[0]
        scopes = {'edited_core': spatial[0] == 1, 'person': people[0], 'neutral_person': neutral}
        bounds = {key: float(floor[mask].mean()) if mask.any() else None for key, mask in scopes.items()}
        cases[pair] = dict(bounds=bounds, score=score(output, authored, target, people, spatial), outputs=renders)
    source = Path(__file__)
    report = dict(cases=cases, prior_sha256=PARENT_REPORT, colorfm_evidence_sha256=[entry[2] for entry in VARIANTS],
                  source_sha256=digest(source), protocol_sha256=digest(source.with_name(
                      'colorfm-response-mixture-bound-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: dict(bounds=row['bounds'], checks=row['score']['gates']['checks'])
                      for key, row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
