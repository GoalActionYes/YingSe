"""Compare uniform and target-specific cubic fits under known look transforms."""
import argparse
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_aligned_cubic_pilot import cubic_mapping, learn
from probe_arri_shared_look import metrics
from probe_bg_real_film_b_qualification import transform
from probe_bg_real_paired_two_look import SWITCH_SHA
from probe_capture_invariant_look import style_transforms
from probe_paired_cubic_cross_content import apply
from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, json_to, panel, save
from spatial_render import finish_spatial
from verify_development_inputs import manifest

PARENT = 'bf7ae001333b56640c047f31a9318e5b9e66856fc1d218e5b5a3de36cb52ee54'


def fit(pixels, styled, selected):
    weights = learn(pixels, styled, selected.ravel())
    if weights is None:
        raise ValueError('Insufficient known color support')
    return cubic_mapping(weights)


def uniform_tables(styles):
    axis = np.rint(np.linspace(0, 255, 24)).astype(np.uint8)
    chart = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(96, 144, 3)
    return {label: fit(chart, transform(chart, processor)[0], np.ones(chart.shape[:2], bool))
            for label, processor in styles.items()}


def readings(a, b, off, on, areas, support):
    output = {}
    for label, region in areas.items():
        mask = region & support
        pixels = int(mask.sum())
        if pixels < 8:
            output[label] = {'pixels': pixels, 'score': None}
            continue
        value = metrics(a, b, off, on, mask)
        signal = value['original_matches_authored_mae']
        output[label] = {'pixels': pixels, 'score': value,
                         'eligible': signal >= 1 and (label != '白球服' or pixels >= 40)}
    return output


def measure(out, key, row, archive, historical, old_pair, styles, common):
    anchor = BASE / 'bg-real-editor-look' / key
    source = checked_rgb(anchor / 'target.png', row['artifacts']['target.png'])
    looked = {label: transform(source, processor)[0] for label, processor in styles.items()}
    a, b = [finish_spatial(source, looked[label], STRENGTH) for label in ('film_a','film_b')]
    prior = archive['results']['13'][key]
    np.testing.assert_array_equal(a, checked_rgb(BASE / 'bg-real-two-look-switch-v3' / key / 'expected-a.png',
                                                   prior['asset_hashes']['expected-a']))
    np.testing.assert_array_equal(b, checked_rgb(BASE / 'bg-real-film-b-qualification-v1' / f'{key}-expected.png',
                                                   historical['qualification'][key]['expected_sha256']))
    y, x = np.indices(source.shape[:2])
    train = (x // 16 + y // 16) % 2 == 0
    whole = np.ones(train.shape, dtype=bool)
    neutral = np.logical_and.reduce([target_regions(v, ~whole)['neutral_proxy'] for v in (source, a, b)])
    areas = {'whole': whole, 'neutral': neutral}
    for name, rect in regions()[key].items():
        mask = np.zeros_like(train)
        mask[rect] = True
        areas[name] = mask
    local = {label: fit(source, frame, train) for label, frame in looked.items()}
    rows, boards = {}, {}
    out.mkdir()
    for arm, tables in (('uniform', common), ('target_half', local)):
        left, right = [apply(source, tables[label]) for label in ('film_a','film_b')]
        hashes = {style: save(out / f'{arm}-{style}.png', rgb) for style, rgb in (('a', left), ('b', right))}
        rows[arm] = dict(holdout=readings(a,b,left,right,areas,~train), hashes=hashes)
        if arm == 'uniform':
            rows[arm]['whole_image'] = readings(a,b,left,right,areas,whole)
        boards[arm] = {'target': source, 'A': left, 'B': right, 'known A': a, 'known B': b}
    for role, receipt in old_pair['cases'][key]['roles'].items():
        left = checked_rgb(BASE / 'bg-real-paired-two-look-v1' / key / f'{role}-a.png', receipt['frame_a_sha256'])
        right = checked_rgb(BASE / 'bg-real-film-b-qualification-v1' / f'paired-{key}-{role}-on.png',
                            historical['comparison']['paired']['cases'][key]['outputs'][f'{role}-on'])
        rows[role] = {'holdout': readings(a,b,left,right,areas,~train)}
    return dict(arms=rows, panel_sha256=panel(out / 'comparison.png', boards))


def run(out):
    archive = checked_json(BASE / 'bg-real-two-look-switch-v3/results.json', SWITCH_SHA)
    historical = checked_json(BASE / 'bg-real-film-b-qualification-v1/results.json', archive['film_b_report_sha256'])
    old_pair = checked_json(BASE / 'bg-real-paired-two-look-v1/results.json', PARENT)
    styles = style_transforms(BASE)
    tables = uniform_tables(styles)
    out.mkdir()
    cases = {key: measure(out / key, key, row, archive, historical, old_pair, styles, tables)
             for key, row in manifest(BASE, 'bg-real-editor-look')['cases'].items()}
    json_to(out / 'results.json', dict(cases=cases, parent_sha256=PARENT, holdout='16px checkerboard'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
