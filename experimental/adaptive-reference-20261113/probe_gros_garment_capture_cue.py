"""Compare paired material changes before replaying a synthetic portrait edit."""
import argparse
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np

from probe_beyond_rgb_matched_cast import compare
from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_gros_screen_chroma import ARCHIVE, face_weight, local_edit, local_score
from probe_multilum_gray_probe import apply, chroma
from probe_playraw_mixed_light_transfer import PARENT, PARENT_SHA, read
from probe_target_mirror_chroma import STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import finish_spatial


LOCAL = 'c057e8ec8789531a5df8dfe0689994b28704ed6a64ee1008efb170106e3b5547'
STATES = ('cold', 'warm', 'matched', 'face_only')


def rectangles(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError('Region outside image')
        mask[y0:y1, x0:x1] = True
    return mask


def sample(image, zone):
    data = image[zone]
    usable = data[((data >= 8) & (data <= 247)).all(axis=-1)]
    return (np.median(chroma(usable), axis=0) if len(usable) >= 256 else None), len(usable)


def estimate(before, after, prior_zones, after_zones):
    changes = {}
    for label in prior_zones:
        a, count_a = sample(before, prior_zones[label])
        b, count_b = sample(after, after_zones[label])
        delta = b - a if a is not None and b is not None else None
        changes[label] = dict(delta=None if delta is None else delta.tolist(), samples=[count_a, count_b])
    background = changes['background']['delta']
    face = changes['face']['delta']
    shirt = changes['shirt']['delta']
    result = dict(changes=changes, route='uncertain', face_residual=None, shirt_residual=None, agreement=None)
    if any(value is None for value in (background, face, shirt)):
        return result
    residual_face = np.array(face) - background
    residual_shirt = np.array(shirt) - background
    size_face, size_shirt = (float(np.linalg.norm(value)) for value in (residual_face, residual_shirt))
    result.update(face_residual=residual_face.tolist(), shirt_residual=residual_shirt.tolist(),
                  face_residual_norm=size_face, shirt_residual_norm=size_shirt)
    if size_face >= .10 and size_shirt >= .10:
        agreement = float((residual_face @ residual_shirt) / (size_face * size_shirt))
        result['agreement'] = agreement
        if agreement >= .50:
            result['route'] = 'capture'
    elif size_shirt < .10:
        result['route'] = 'local' if size_face >= .10 else 'global'
    return result


def authored(boxes):
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    off, _ = read(PARENT / 'off.tif', False)
    on, weight = read(PARENT / 'on.tif', True)
    if ([sha256(rgb.tobytes()).hexdigest() for rgb in (off, on)] != parent['rgb_sha256']
            or sha256(weight.tobytes()).hexdigest() != parent['mask_pixels_sha256']):
        raise ValueError('Author RGB or mask changed')
    zones = {label: rectangles(off.shape[:2], coords) for label, coords in boxes.items()}
    actual = estimate(off, on, zones, zones)
    actual['author_high_weight_pixels'] = {label: int((weight[area] >= 204).sum()) for label, area in zones.items()}
    actual['qualified'] = (actual['author_high_weight_pixels']['face'] >= 128
                           and not any(weight[zones[name]].any() for name in ('shirt', 'background')))
    return actual


def render(target, weight, evidence, clothes):
    delta = evidence['changes']['background']['delta']
    if delta is None or (clothes and evidence['route'] == 'capture'):
        return target.copy()
    shifted, _ = apply(target, np.asarray(delta))
    if clothes and evidence['route'] == 'local':
        shifted = local_edit(shifted, np.asarray(evidence['face_residual']), weight)
    return finish_spatial(target, shifted, STRENGTH)


def run(gros, scope, out, *, estimate_material=estimate, additions=None):
    source = checked_json(gros / 'results.json', ARCHIVE)
    previous = checked_json(scope / 'results.json', LOCAL)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    garment_file = Path(__file__).with_name('gros-garment-regions.json')
    if digest(region_file) != source['regions_sha256'] or previous['archive_sha256'] != ARCHIVE:
        raise ValueError('Changed reference archive or face geometry')
    regions = json.loads(region_file.read_bytes())
    garments = json.loads(garment_file.read_bytes())
    guides, target_regions, colors, weights = {}, {}, {}, {}
    for group in GROUPS:
        admission = source['admission'][group]
        guides[group] = {state: checked_rgb(gros / 'guides' / f'{group}-{state}.png', sha)
                         for state, sha in admission['guide_sha256'].items()}
        guides[group]['face_only'] = checked_rgb(scope / 'local-guides' / f'{group}.png',
                                                 previous['local_guides'][group])
        shape = guides[group]['cold'].shape[:2]
        materials = {}
        for state in ('cold', 'warm'):
            selected = masks(regions[group][state], regions['screen'], shape)
            materials[state] = {'face': selected['cheek_left'] | selected['cheek_right'],
                                'background': selected['screen_left'] | selected['screen_right'],
                                'shirt': rectangles(shape, garments['gros'])}
            if state == 'cold':
                target_regions[group] = selected
        colors[group] = {
            state: estimate_material(guides[group]['cold'], guides[group][state], materials['cold'],
                                     materials['warm' if state == 'warm' else 'cold']) for state in STATES
        }
        weights[group] = face_weight(guides[group]['cold'], regions[group]['cold']['eyes'])

    author = authored(garments['authored'])
    out.mkdir()
    results = {}
    for target, donor in permutations(GROUPS, 2):
        name = f'{target}-from-{donor}'
        folder = out / name
        folder.mkdir()
        original = guides[target]['cold']
        expected_global = checked_rgb(gros / name / 'expected.png', source['results'][name]['expected_sha256'])
        expected_local = checked_rgb(scope / name / 'expected-local.png', previous['cases'][name]['expected_local_sha256'])
        board, rows = {'reference': guides[donor]}, {}
        for label, clothes in (('background', False), ('garment', True)):
            frames = {state: render(original, weights[target], colors[donor][state], clothes) for state in STATES}
            global_scores = {region: compare(original, expected_global, frames['cold'], frames['warm'],
                                             frames['matched'], mask) for region, mask in target_regions[target].items()}
            local = local_score(original, expected_local, frames, target_regions[target], weights[target])
            identity = bool(np.array_equal(frames['cold'], original))
            known_paths = all(colors[donor][state]['route'] != 'uncertain' for state in STATES) if clothes else True
            joint = (identity and known_paths and local['passed'] and all(row['both_pass'] for row in global_scores.values()))
            rows[label] = dict(global_scores=global_scores, local_scores=local, identity=identity,
                               uncertain_route=not known_paths, joint=joint,
                               hashes={state: save(folder / f'{label}-{state}.png', frame) for state, frame in frames.items()})
            board[label] = frames
        board['target'] = {'original': original, 'expected global': expected_global, 'expected local': expected_local}
        results[name] = dict(arms=rows, panel_sha256=panel(folder / 'comparison.png', board))
    count = {label: sum(row['arms'][label]['joint'] for row in results.values()) for label in ('background', 'garment')}
    passed = count['garment'] >= 5 and author['qualified'] and author['route'] == 'local'
    json_to(out / 'results.json', dict(gros_sha256=ARCHIVE, local_sha256=LOCAL, script_sha256=digest(Path(__file__)),
                                      garment_regions_sha256=digest(garment_file), regions_sha256=digest(region_file),
                                      donor_measurements=colors, author=author, cases=results, counts=count,
                                      passed=passed, **(additions or {})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gros', 'scope', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
