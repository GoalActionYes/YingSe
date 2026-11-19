"""Exercise the ordinary workbench with a photographed local-edit switch."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_frozen_candidate_development_crosscheck import CHECKPOINT, STRENGTH
from probe_playraw_mixed_light_skin import read
from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, save
from spatial_render import SpatialEngine, finish_spatial
from spatial_store import SpatialStore
from verify_development_inputs import manifest


PARENT = BASE / 'playraw-mixed-light-skin-v1/replay-v2'
PARENT_SHA = 'b5717f8adb5920abfaa1f8be6e207875aa519b90b4f161a6b61d1b73418e82d9'
WHITE = '872-1268'
LUMA = np.array([.2126, .7152, .0722])


def chroma(rgb):
    rgb = rgb.astype(np.float64)
    return rgb[..., [0, 2]] - (rgb @ LUMA)[..., None]


def stats(actual, desired, region):
    change, want = actual[region], desired[region]
    error = np.abs(change - want).mean(axis=1)
    energy = np.sum(want ** 2)
    return {'pixels': len(change), 'projection': float(np.sum(change * want) / energy) if energy else None,
            'response_mae': float(error.mean()), 'response_p95': float(np.quantile(error, .95))}


def render(store, target, references, out, name):
    target_id, target_rgb = target
    rendered, records = [], {}
    for state, (image, reference_id) in references.items():
        mapping = store.analyze(target_id, reference_id)
        ident = mapping['id']
        diagnostic = mapping['diagnostics']
        frame = store.render(ident, target_rgb, STRENGTH)
        np.testing.assert_array_equal(frame, store.render(ident, target_rgb, STRENGTH, preview=True))
        if np.array_equal(image, target_rgb):
            np.testing.assert_array_equal(frame, target_rgb)
        records[state] = {'image_sha256': save(out / f'{name}-{state}.png', frame),
                          'identity': diagnostic['exact_identity'],
                          'aligned': diagnostic['certified_aligned']['accepted']}
        rendered.append(chroma(frame))
    return rendered[1] - rendered[0], records


def register(store, out, name, image):
    path = out / f'input-{name}.png'
    save(path, image)
    entry = store.add_media(path)
    decoded, _ = store.decode(entry)
    np.testing.assert_array_equal(decoded, image)
    return entry['id'], decoded


def run(out):
    torch.set_num_threads(4)
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    before, _ = read(PARENT / 'off.tif', False)
    after, mask = read(PARENT / 'on.tif', True)
    if ([sha256(x.tobytes()).hexdigest() for x in (before, after)] != parent['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != parent['mask_pixels_sha256']):
        raise ValueError('Changed local edit source')
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 checkpoint')
    out.mkdir()
    store = SpatialStore(out / 'store', engine)
    target = register(store, out, 'target', before)
    expected = chroma(finish_spatial(before, after, STRENGTH)) - chroma(before)
    neutral = target_regions(before, np.zeros(mask.shape, bool))['neutral_proxy']
    zones = {'author': mask >= 204, 'protected': mask == 0, 'neutral': neutral & (mask == 0)}
    h, w = mask.shape
    crop = np.s_[h//16:h-h//16, w//16:w-w//16]
    coverage = float(zones['author'][crop].sum() / zones['author'].sum())
    pairs, cases = {}, {}
    for name, views in {'full': (before, after), 'mirror': (before[:, ::-1], after[:, ::-1]),
                        'crop': (before[crop], after[crop])}.items():
        pairs[name] = {state: (rgb, register(store, out, f'{name}-{state}', rgb)[0])
                       for state, rgb in zip(('off', 'on'), views, strict=True)}
        change, records = render(store, target, pairs[name], out, name)
        data = {label: stats(change, expected if label == 'author' else np.zeros_like(expected), zone)
                for label, zone in zones.items()}
        eligible = name != 'crop' or coverage >= .95
        passed = (.75 <= data['author']['projection'] <= 1.25 and data['author']['response_mae'] <= 3
                  and max(data[label]['response_p95'] for label in ('protected', 'neutral')) <= 1.5)
        cases[name] = {'outputs': records, 'scores': data, 'qualified': eligible, 'passed': passed if eligible else None}
    info = manifest(BASE, 'bg-real-editor-look')['cases'][WHITE]
    white = checked_rgb(BASE / 'bg-real-editor-look' / WHITE / 'target.png', info['artifacts']['target.png'])
    white_target = register(store, out, 'white-target', white)
    response, records = render(store, white_target, pairs['full'], out, 'white')
    jersey = np.zeros(white.shape[:2], bool)
    jersey[regions()[WHITE]['白球服']] = True
    white_scores = stats(response, np.zeros_like(response), jersey)
    white_pass = white_scores['response_p95'] <= 1.5
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'cases': cases, 'crop_coverage': coverage,
                                   'white': {'outputs': records, 'scores': white_scores, 'passed': white_pass},
                                   'passed': white_pass and all(row['passed'] for row in cases.values() if row['qualified'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
