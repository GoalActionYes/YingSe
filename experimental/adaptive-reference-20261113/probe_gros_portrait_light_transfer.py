"""Render two baseline pipelines on matched real and digital portrait casts."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch

from probe_beyond_rgb_matched_cast import compare
from probe_multilum_gray_probe import apply, chroma, valid
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze
from spatial_render import SpatialEngine, finish_spatial
from spatial_store import SpatialStore
from triplet_color import to_srgb


PHOTO_SHA = '927171d871581440af236eab428bae08f9bcc391ac4028582376753d824b0a4f'
GROUPS = ('01', '02', '03')


def load_photo(path, expected):
    if digest(path) != expected:
        raise ValueError('Changed Gros photo')
    with Image.open(path) as image:
        rgb, evidence = to_srgb(image)
        return np.array(rgb.resize((768, 512), Image.Resampling.LANCZOS)), evidence['policy']


def masks(record, white, shape):
    selections = {'whole': np.ones(shape, dtype=bool)}
    boxes = {name: record[name] for name in ('cheek_left', 'cheek_right')} | white
    for name, (x0, y0, x1, y1) in boxes.items():
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError('Material region outside image')
        area = np.zeros(shape, dtype=bool)
        area[y0:y1, x0:x1] = True
        if area.sum() < 128:
            raise ValueError('Insufficient material samples')
        selections[name] = area
    return selections


def fit_guides(images, areas, eyes):
    cold, warm = (images[state] for state in ('cold', 'warm'))
    geometry = np.linalg.norm(eyes['warm'] - eyes['cold'], axis=1)
    ratio = np.linalg.norm(np.diff(eyes['warm'], axis=0)) / np.linalg.norm(np.diff(eyes['cold'], axis=0))

    def sample(image, regions):
        support = valid(image)
        cheek = regions['cheek_left'] | regions['cheek_right']
        region_masks = {'cheeks': cheek, 'screen_left': regions['screen_left'], 'screen_right': regions['screen_right']}
        return {name: (np.median(chroma(image[area & support]), axis=0), int((area & support).sum()))
                for name, area in region_masks.items()}

    a, b = (sample(images[state], areas[state]) for state in ('cold', 'warm'))
    delta = b['cheeks'][0] - a['cheeks'][0]
    edited, clipped = apply(cold, delta)
    c = sample(edited, areas['cold'])
    movements = {name: dict(real=(b[name][0] - a[name][0]).tolist(),
                            digital=(c[name][0] - a[name][0]).tolist(),
                            samples=[a[name][1], b[name][1], c[name][1]]) for name in a}
    magnitude = float(np.linalg.norm(delta))
    difference = float(np.linalg.norm(c['cheeks'][0] - b['cheeks'][0]))
    qualified = (max(geometry) <= 20 and .90 <= ratio <= 1.10
                 and magnitude >= .08 and difference <= .025
                 and min(movements['cheeks']['samples']) >= 256 and clipped <= .02)
    return dict(cold=cold, warm=warm, matched=edited), delta, dict(
        eye_shift=geometry.tolist(), eye_spacing_ratio=float(ratio), chroma_strength=magnitude,
        match_error=difference, clipped=clipped, regions=movements, qualified=bool(qualified))


@torch.inference_mode()
def run(source, out):
    original = checked_json(source / 'report.json', PHOTO_SHA)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    coordinates = json.loads(region_file.read_bytes())
    out.mkdir()
    guide_folder = out / 'guides'
    guide_folder.mkdir()
    guides, shifts, admission, regions = {}, {}, {}, {}
    for group in GROUPS:
        images, color_management, selections, eyes = {}, {}, {}, {}
        for state in ('cold', 'warm'):
            key = f'{group}-{state}'
            images[state], color_management[state] = load_photo(source / f'{key}.jpg', original['photos'][key]['sha256'])
            box = coordinates[group][state]
            selections[state] = masks(box, coordinates['screen'], images[state].shape[:2])
            eyes[state] = np.array(box['eyes'])
        variants, shifts[group], audit = fit_guides(images, selections, eyes)
        guides[group] = variants
        regions[group] = selections['cold']
        hashes = {state: save(guide_folder / f'{group}-{state}.png', rgb) for state, rgb in variants.items()}
        admission[group] = {**audit, 'input_color_policy': color_management, 'guide_sha256': hashes}

    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Baseline model changed')
    rows, identities = {}, {}
    with TemporaryDirectory(prefix='gros-portrait-spatial-') as workspace:
        store = SpatialStore(Path(workspace), engine)
        assets = {group: {state: store.add_media(guide_folder / f'{group}-{state}.png')['id']
                          for state in frames} for group, frames in guides.items()}
        for group in GROUPS:
            rgb = guides[group]['cold']
            path = guide_folder / f'{group}-cold.png'
            stable = apply_mapping(rgb, analyze(path, path), STRENGTH)
            entry = store.analyze(assets[group]['cold'], assets[group]['cold'])
            spatial = store.render(entry['id'], rgb, STRENGTH)
            identities[group] = dict(stable=bool(np.array_equal(stable, rgb)), spatial=bool(np.array_equal(spatial, rgb)))
        for target, donor in permutations(GROUPS, 2):
            name = f'{target}-from-{donor}'
            folder = out / name
            folder.mkdir()
            original_rgb = guides[target]['cold']
            expected = finish_spatial(original_rgb, apply(original_rgb, shifts[donor])[0], STRENGTH)
            baseline = {'stable': {}, 'spatial13': {}}
            routes = {}
            for state in guides[donor]:
                model = analyze(guide_folder / f'{target}-cold.png', guide_folder / f'{donor}-{state}.png')
                baseline['stable'][state] = apply_mapping(original_rgb, model, STRENGTH)
                entry = store.analyze(assets[target]['cold'], assets[donor][state])
                baseline['spatial13'][state] = store.render(entry['id'], original_rgb, STRENGTH)
                routes[state] = entry['diagnostics']['certified_aligned']['accepted']
            board = {'inputs': {'target': original_rgb, **guides[donor], 'expected edit': expected}}
            layers = {}
            for arm, frames in baseline.items():
                scores = {region: compare(original_rgb, expected, *frames.values(), area)
                          for region, area in regions[target].items()}
                hashes = {state: save(folder / f'{arm}-{state}.png', image) for state, image in frames.items()}
                layers[arm] = dict(scores=scores, passed=admission[donor]['qualified'] and
                                   all(value['both_pass'] for value in scores.values()), hashes=hashes)
                board[arm] = frames
            rows[name] = dict(reference_qualified=admission[donor]['qualified'], certified_affine=routes, layers=layers,
                              expected_sha256=save(folder / 'expected.png', expected),
                              panel_sha256=panel(folder / 'comparison.png', board))
    json_to(out / 'results.json', dict(input_sha256=PHOTO_SHA, regions_sha256=digest(region_file),
                                      script_sha256=digest(Path(__file__)), checkpoint_sha256=engine.checkpoint_sha,
                                      stable_fingerprint=algorithm_fingerprint(), strength=STRENGTH,
                                      admission=admission, identity=identities, results=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
