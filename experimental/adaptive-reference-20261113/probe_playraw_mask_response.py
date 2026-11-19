"""Check authored mask weighting against an independently rendered uniform edit."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import re

import numpy as np

from audit_playraw_sky_xmp import DT, payload
from probe_hdm_hdr_display import json_to
from probe_playraw_mixed_light_cross_editor import FIT_SHA, INPUTS, step
from probe_playraw_mixed_light_skin import INPUT_HASHES, export, read
from probe_playraw_mixed_light_transfer import PARENT_SHA, STRENGTH, chroma, stats
from probe_target_mirror_chroma import checked_json, save
from spatial_render import finish_spatial


def uniform(source):
    before = step(source, '28')
    old = before[DT + 'blendop_params']
    value = payload(old)
    if value[:4] != b'\x03\x00\x00\x00' or before[DT + 'enabled'] != '1':
        raise ValueError('Expected active drawn mask')
    new = (b'\x01\x00\x00\x00' + value[4:]).hex()
    tag, = [item for item in re.finditer(rb'<rdf:li\s+[^>]*>', source) if b'darktable:num="28"' in item[0]]
    old_attr = f'darktable:blendop_params="{old}"'.encode()
    new_attr = f'darktable:blendop_params="{new}"'.encode()
    if tag[0].count(old_attr) != 1:
        raise ValueError('Unexpected mask setting')
    state = source[:tag.start()] + tag[0].replace(old_attr, new_attr) + source[tag.end():]
    if step(state, '28') != before | {DT + 'blendop_params': new}:
        raise ValueError('Unexpected edit differences')
    return state


def evaluate(base, local, mask, complete, folds, output):
    old = chroma(finish_spatial(base, local, STRENGTH)) - chroma(base)
    full = chroma(finish_spatial(base, complete, STRENGTH)) - chroma(base)
    estimate = full * (mask.astype(np.float64) / 255)[..., None]
    areas = {'full': mask == 255, 'high': (mask >= 204) & (mask < 255),
             'middle': (mask >= 64) & (mask < 204), 'low': (mask > 0) & (mask < 64), 'zero': mask == 0}
    rows = {name: stats(estimate, old, area) if area.any() else {'pixels': 0} for name, area in areas.items()}
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    corrections = []
    for i in range(2):
        sampled = np.median(full[(mask >= 204) & (parity == i)], axis=0)
        fitted = np.asarray(folds[str(i)]['constant'])
        corrections.append({'uniform': sampled.tolist(), 'normalized_local': fitted.tolist(),
                            'difference': float(np.abs(sampled - fitted).mean())})
    controls = (rows['full']['pixels'] >= 128 and rows['full']['response_mae'] <= 1
                and rows['zero']['response_p95'] <= 1.5)
    return {'areas': rows, 'corrected_chroma': corrections, 'controls': controls,
            'material_high_weight_error': controls and all(row['difference'] >= 1 for row in corrections),
            'image_sha256': save(output, complete)}


def run(root, out):
    if not all(p.is_absolute() and p.as_posix().isascii() for p in (root, out)):
        raise ValueError('Use absolute ASCII paths')
    folder = root / 'playraw-mixed-light-skin-v1'
    cli, xmp = (root / INPUTS[key][0] for key in ('cli', 'author'))
    raw = folder / 'P5020458.ORF'
    for path in (cli, xmp, raw):
        if sha256(path.read_bytes()).hexdigest() != INPUT_HASHES[path.name]:
            raise ValueError(f'Changed source: {path.name}')
    original = checked_json(folder / 'replay-v2/results.json', PARENT_SHA)
    base, _ = read(folder / 'replay-v2/off.tif', False)
    local, mask = read(folder / 'replay-v2/on.tif', True)
    if ([sha256(a.tobytes()).hexdigest() for a in (base, local)] != original['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != original['mask_pixels_sha256']):
        raise ValueError('Changed author pair')
    folds = checked_json(root / 'playraw-mixed-light-operator-v1/results.json', FIT_SHA)['cases']
    sidecar_data = uniform(xmp.read_bytes())
    editor_env = {key: value for key, value in os.environ.items() if key != 'OMP_NUM_THREADS'}
    out.mkdir()
    sidecar = out / 'uniform.xmp'
    sidecar.write_bytes(sidecar_data)
    rows, images = [], []
    for i in range(2):
        label = f'uniform-{i}'
        rgb, _ = read(export(cli, raw, sidecar, out, label, False, env=editor_env), False)
        images.append(rgb)
        rows.append(evaluate(base, local, mask, rgb, folds, out / f'{label}.png'))
    max_delta = int(np.abs(images[0].astype(np.int16) - images[1]).max())
    json_to(out / 'results.json', {'trials': rows, 'sidecar_sha256': sha256(sidecar_data).hexdigest(),
                                  'max_repeat_delta': max_delta, 'parents': [PARENT_SHA, FIT_SHA],
                                  'material_high_weight_error': max_delta <= 1 and all(
                                      row['material_high_weight_error'] for row in rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
