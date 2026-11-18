"""Fit a depth-independent spatial offset to a known local edit offline."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import grid_sample

from background_intervention import edit_weight, target_regions
from independent_coco import digest
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import COOL, ROLES, compare, gates_for, mask_for, save, versions
from probe_reference_chroma_spread import BASE, PARENT
from probe_roundtrip_grade_choice import checked_json, read_rgb
from spatial_render import coordinates, finish_spatial


def basis(length, knots):
    """Linear basis at pixel centers with border padding, align_corners=False."""
    position = np.clip((np.arange(length) + .5) * knots / length - .5, 0, knots - 1)
    return np.maximum(1 - np.abs(position[:, None] - np.arange(knots)), 0)


def project(weight):
    b = basis(256, 16)
    inverse = np.linalg.pinv(b)
    nodes = inverse @ weight @ inverse.T
    guide = torch.full((1, 256, 256), .5, dtype=torch.float64)
    coords = coordinates(256, 256, 0, 256, guide)
    grid = torch.from_numpy(nodes)[None, None, None].expand(1, 1, 8, 16, 16).contiguous()
    projected = grid_sample(grid, coords, align_corners=False, padding_mode='border')[0, 0, 0].numpy()
    np.testing.assert_allclose(projected, b @ nodes @ b.T, atol=1e-12, rtol=0)
    return projected


def render(target, edits, projected, has_person):
    result = {role: finish_spatial(target, rgb, .8) for role, rgb in edits.items()}
    if projected is not None:
        base = 127.5 + .88 * (target.astype(np.float64) - 127.5) + np.array([6., 2., -5.])
        raw = base + projected[..., None] * COOL
        result['local'] = finish_spatial(target, np.rint(raw.clip(0, 255)).astype(np.uint8), .8)
        if not has_person:
            np.testing.assert_array_equal(result['local'], result['base'])
    return result


def score(images, authored, target, people, weights):
    person = people[0]
    weight = weights[0]
    masks = target_regions(target, person)
    masks.update(edited_core=weight == 1, untouched=weight == 0,
                 neutral_person=masks['neutral_proxy'] & person,
                 transition=(weight > 0) & (weight < 1))
    stats = compare(images, authored, masks)
    values = {
        f'{role}_{region}': None if entry is None else {
            'error': entry['error'], 'projection': entry['projection']
        }
        for role, regions in stats.items() for region, entry in regions.items()
    }
    return dict(gates=gates_for(stats, images, people, weights), measures=values)


def run(out):
    start = time.monotonic()
    torch.set_num_threads(2)
    docs = Path(__file__).resolve().parents[2] / 'docs/research'
    frozen = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    parent = checked_json(PARENT / 'results.json', frozen['parent_sha256'])
    out.mkdir(exist_ok=False)
    cases = {}
    for case in parent['cases']:
        pair = case['pair']
        people = tuple(mask_for(case, image_id) for image_id in pair.split('-'))
        weights = tuple(edit_weight(person, 1)[0] if person.any() else np.zeros_like(person, dtype=float)
                        for person in people)
        root = BASE / 'local-edit-toggle-v1' / pair
        files = frozen['cases'][pair]['files']
        target = read_rgb(root / 'target.png', files['target.png'])
        edits, exact, _ = versions(target, people[0])
        np.testing.assert_array_equal(exact, weights[0])
        smooth = project(exact)
        predicted = dict(exact=render(target, edits, None, people[0].any()),
                         grid=render(target, edits, smooth, people[0].any()))
        paths = {mode: {role: save(out / f'{pair}-{mode}-{role}.png', rgb) for role, rgb in images.items()}
                 for mode, images in predicted.items()}
        authored = {role: read_rgb(root / f'{role}-expected.png', files[f'{role}-expected.png']) for role in ROLES}
        for role in ROLES:
            np.testing.assert_array_equal(predicted['exact'][role], authored[role])
        predicted['current13'] = {
            role: read_rgb(root / f'{role}-current13.png', files[f'{role}-current13.png']) for role in ROLES
        }
        cases[pair] = dict(predictions=paths, scores={
            mode: score(images, authored, target, people, weights) for mode, images in predicted.items()
        }, projected_range=[float(smooth.min()), float(smooth.max())],
            max_protected_weight=float(np.abs(smooth[exact == 0]).max()))
    elapsed = time.monotonic() - start
    if elapsed > 45:
        raise TimeoutError('Frozen local capacity budget exceeded')
    source = Path(__file__)
    result = dict(cases=cases, elapsed_seconds=elapsed, author_evidence_sha256=PARENT_REPORT,
                  source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('local-grid-capacity-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
