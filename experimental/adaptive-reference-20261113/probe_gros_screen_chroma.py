"""Check a known neutral material for separating capture spill and edit intent."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from probe_arri_shared_look import metrics
from probe_beyond_rgb_matched_cast import compare
from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_multilum_gray_probe import apply, chroma, valid
from probe_physical_blue_global_fit import display, linear
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import SpatialEngine, finish_spatial
from spatial_store import SpatialStore


ARCHIVE = 'a23b304c4423e908cf68ff0bc6612c0107e1464578d460b1dfcd0a74f837a664'


def white_chroma(image, regions):
    mask = (regions['screen_left'] | regions['screen_right']) & valid(image)
    if mask.sum() < 256:
        raise ValueError('Insufficient neutral material')
    return np.median(chroma(image[mask]), axis=0)


def face_weight(image, eyes):
    eyes = np.asarray(eyes)
    x, y = eyes.mean(axis=0)
    spacing = np.linalg.norm(eyes[1] - eyes[0])
    yy, xx = np.indices(image.shape[:2])
    radius = np.hypot((xx - x) / (1.25 * spacing), (yy - y - .42 * spacing) / (1.10 * spacing))
    return np.clip((1 - radius) / .18, 0, 1)


def local_edit(image, delta, weight):
    graded, _ = apply(image, delta)
    source = linear(image)
    mixed = source + weight[..., None] * (linear(graded) - source)
    return display(mixed)


def grade(target, source_chroma, reference_chroma):
    candidate, clipped = apply(target, reference_chroma - source_chroma)
    return finish_spatial(target, candidate, STRENGTH), clipped


def local_score(target, expected, frames, regions, weight):
    off, on = frames['cold'], frames['face_only']
    measurements = {name: metrics(target, expected, off, on, regions[name])
                    for name in ('cheek_left', 'cheek_right')}
    actual = np.abs(on.astype(np.int16) - off)
    outside = float(actual[weight == 0].mean())
    passed = outside <= 1.5 and all(values['passed'] for values in measurements.values())
    return dict(cheeks=measurements, outside_mae=outside, passed=passed)


@torch.inference_mode()
def run(archive, out):
    parent = checked_json(archive / 'results.json', ARCHIVE)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    if digest(region_file) != parent['regions_sha256']:
        raise ValueError('Changed material registration')
    coordinates = json.loads(region_file.read_bytes())
    out.mkdir()
    reference_folder = out / 'local-guides'
    reference_folder.mkdir()
    guides, regions, weights, offsets, white, local_hashes, support = {}, {}, {}, {}, {}, {}, {}
    for group in GROUPS:
        evidence = parent['admission'][group]
        guides[group] = {state: checked_rgb(archive / 'guides' / f'{group}-{state}.png', sha)
                         for state, sha in evidence['guide_sha256'].items()}
        base = guides[group]['cold']
        weight = face_weight(base, coordinates[group]['cold']['eyes'])
        weights[group] = weight
        offsets[group] = np.array(evidence['regions']['cheeks']['real'])
        guides[group]['face_only'] = local_edit(base, offsets[group], weight)
        regions[group] = masks(coordinates[group]['cold'], coordinates['screen'], base.shape[:2])
        support[group] = {name: float(np.median(weight[regions[group][name]]))
                          for name in ('cheek_left', 'cheek_right')}
        neutral_region = regions[group]['screen_left'] | regions[group]['screen_right']
        np.testing.assert_array_equal(guides[group]['face_only'][neutral_region], base[neutral_region])
        white[group] = {state: white_chroma(image, regions[group]) for state, image in guides[group].items()}
        local_hashes[group] = save(reference_folder / f'{group}.png', guides[group]['face_only'])
    identity = {group: bool(np.array_equal(grade(guides[group]['cold'], white[group]['cold'], white[group]['cold'])[0],
                                          guides[group]['cold'])) for group in GROUPS}

    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed baseline model')
    outputs = {}
    with TemporaryDirectory(prefix='gros-screen-chroma-') as workspace:
        store = SpatialStore(Path(workspace), engine)
        targets = {key: store.add_media(archive / 'guides' / f'{key}-cold.png')['id'] for key in GROUPS}
        references = {key: store.add_media(reference_folder / f'{key}.png')['id'] for key in GROUPS}
        for target, donor in permutations(GROUPS, 2):
            key = f'{target}-from-{donor}'
            folder = out / key
            folder.mkdir()
            original = guides[target]['cold']
            regions_target = regions[target]
            prior = parent['results'][key]
            old = prior['layers']['spatial13']
            baseline = {state: checked_rgb(archive / key / f'spatial13-{state}.png', sha)
                        for state, sha in old['hashes'].items()}
            entry = store.analyze(targets[target], references[donor])
            baseline['face_only'] = store.render(entry['id'], original, STRENGTH)
            grade_outputs = {state: grade(original, white[target]['cold'], measured)
                             for state, measured in white[donor].items()}
            candidate = {state: value[0] for state, value in grade_outputs.items()}
            expected_global = checked_rgb(archive / key / 'expected.png', prior['expected_sha256'])
            expected_local = finish_spatial(original, local_edit(original, offsets[donor], weights[target]), STRENGTH)
            rows = {}
            balanced = all(value >= .90 for group in (target, donor) for value in support[group].values())
            for name, frames in (('spatial13', baseline), ('known_screen', candidate)):
                main = {label: compare(original, expected_global, frames['cold'], frames['warm'], frames['matched'], area)
                        for label, area in regions_target.items()}
                if name == 'spatial13' and main != old['scores']:
                    raise ValueError('Frozen cross-portrait score changed')
                local = local_score(original, expected_local, frames, regions_target, weights[target])
                passed_global = prior['reference_qualified'] and all(
                    i['both_pass'] and i['style']['off_matches_original_mae'] <= 3 for i in main.values())
                hashes = {state: save(folder / f'{name}-{state}.png', frame) for state, frame in frames.items()
                          if name == 'known_screen' or state == 'face_only'}
                rows[name] = dict(global_scores=main, global_pass=passed_global, local_scores=local,
                                  joint=passed_global and local['passed'] and balanced, hashes=hashes)
            plot = panel(folder / 'comparison.png', {
                'reference': guides[donor], '13': baseline, 'known screen': candidate,
                'target': {'original': original, 'expected global': expected_global, 'expected local': expected_local},
            })
            outputs[key] = dict(arms=rows, eligible_local=balanced,
                                clipped={state: value[1] for state, value in grade_outputs.items()},
                                expected_local_sha256=save(folder / 'expected-local.png', expected_local),
                                panel_sha256=plot)
    json_to(out / 'results.json', dict(archive_sha256=ARCHIVE, script_sha256=digest(Path(__file__)),
                                      regions_sha256=digest(region_file), local_guides=local_hashes,
                                      identity=identity, cheek_weight_median=support, cases=outputs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
