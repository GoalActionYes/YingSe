"""Measure how Filmic RGB changes a recorded color-balance response."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import re

import numpy as np

from audit_playraw_sky_xmp import DT
from probe_playraw_mixed_light_cross_editor import step
from probe_playraw_mixed_light_skin import export
from probe_playraw_pipeline_mask_attribution import MASK_SHA, SRGB_SHA, hash_pixels
from probe_playraw_rendered_input_operator import (
    SOURCE_SHA, STRENGTH, check_input, checked_json, chroma, digest, json_to, panel, read, save,
)


RAW_SHA = '7b4cc05a958acd8e10a161d1b2faf6cae9d25a37de28550b0680b34c14c38217'


def switch(source, include_color_balance):
    matches = 0

    def replace(tag):
        nonlocal matches
        if b'darktable:operation="filmicrgb"' in tag[0]:
            matches += 1
            return tag[0].replace(b'darktable:enabled="1"', b'darktable:enabled="0"')
        if not include_color_balance and b'darktable:num="28"' in tag[0]:
            return tag[0].replace(b'darktable:enabled="1"', b'darktable:enabled="0"')
        return tag[0]

    result = re.sub(rb'<rdf:li\s+[^>]*>', replace, source)
    flips = sum(a != b for a, b in zip(source, result, strict=True))
    if (matches != 4 or flips != 4 + int(not include_color_balance)
            or step(result, '28')[DT + 'enabled'] != str(int(include_color_balance))):
        raise ValueError('Unexpected filmic module history')
    return result


def run(base, out, index):
    cli = check_input(base, 'cli')
    origin = base / 'playraw-mixed-light-skin-v1'
    raw = origin / 'P5020458.ORF'
    if digest(raw) != RAW_SHA:
        raise ValueError('Changed RAW')
    mask_source = checked_json(origin / 'replay-v2/results.json', SOURCE_SHA)
    raw_off, _ = read(origin / 'replay-v2/off.tif', False)
    raw_local, mask = read(origin / 'replay-v2/on.tif', True)
    if [hash_pixels(raw_off), hash_pixels(raw_local)] != mask_source['rgb_sha256'] or (
            hash_pixels(mask) != mask_source['mask_pixels_sha256']):
        raise ValueError('Changed original or mask')
    archive = base / 'playraw-mask-response-v1'
    reference = checked_json(archive / 'results.json', MASK_SHA)
    source = (archive / 'uniform.xmp').read_bytes()
    if sha256(source).hexdigest() != reference['sidecar_sha256']:
        raise ValueError('Changed uniform settings')
    raw_on, _ = read(archive / f'uniform-{index}.tif', False)
    png = archive / f'uniform-{index}.png'
    if digest(png) != reference['trials'][index]['image_sha256']:
        raise ValueError('Changed cached RAW PNG')
    from PIL import Image
    with Image.open(png) as png_image:
        np.testing.assert_array_equal(raw_on, np.asarray(png_image))
    srgb = base / 'playraw-disable-import-sharpen-v2'
    srgb_receipt = checked_json(srgb / 'results.json', SRGB_SHA)['source']['rgb_sha']
    srgb_on, _ = read(srgb / 'mixed/on.tif', False)
    if hash_pixels(srgb_on) != srgb_receipt['on'] or hash_pixels(raw_off) != srgb_receipt['off']:
        raise ValueError('Changed sRGB endpoints')

    out.mkdir()
    settings, trial = {}, {}
    env = {key: value for key, value in os.environ.items() if key != 'OMP_NUM_THREADS'}
    for label in ('off', 'on'):
        path = out / f'{label}.xmp'
        path.write_bytes(switch(source, label == 'on'))
        settings[label] = digest(path)
        trial[label], _ = read(export(cli, raw, path, out, label, False, env=env), False)
        if trial[label].shape != raw_off.shape:
            raise ValueError('Changed geometry')

    high = mask >= 204
    weight = STRENGTH * mask[..., None] / 255
    expected = (chroma(srgb_on) - chroma(raw_off)) * weight
    control = (chroma(raw_on) - chroma(raw_off)) * weight
    candidate = (chroma(trial['on']) - chroma(trial['off'])) * weight
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    areas = {'full': mask == 255, 'fold0': high & (parity != 0), 'fold1': high & (parity != 1)}

    def clipping(image, region):
        pixels = image[region]
        return float(np.any((pixels <= 1) | (pixels >= 254), axis=1).mean())

    rows = {}
    for key, region in areas.items():
        rows[key] = dict(
            pixels=int(region.sum()), control_mae=float(np.abs((control - expected)[region]).mean()),
            disabled_mae=float(np.abs((candidate - expected)[region]).mean()),
            control_off_clipping=clipping(raw_off, region), control_on_clipping=clipping(raw_on, region),
            disabled_off_clipping=clipping(trial['off'], region), disabled_on_clipping=clipping(trial['on'], region))
    aligned = float(np.corrcoef(np.diff(raw_off.astype(float).mean(2), axis=1).ravel(),
                                np.diff(trial['off'].astype(float).mean(2), axis=1).ravel())[0, 1])
    pass_distance = all(rows[key]['control_mae'] - rows[key]['disabled_mae'] >= 1 for key in ('fold0', 'fold1'))
    pass_clip = all(row[f'disabled_{state}_clipping'] - row[f'control_{state}_clipping'] <= .05
                    for row in rows.values() for state in ('off', 'on'))
    files = {state: save(out / f'{state}.png', frame) for state, frame in trial.items()}
    board = panel(out / 'comparison.png', {'Filmic enabled': {'off': raw_off, 'on': raw_on},
                                          'Filmic disabled': trial})
    json_to(out / 'results.json', dict(index=index, script_sha=digest(Path(__file__)),
            source_hashes=[SOURCE_SHA, MASK_SHA, SRGB_SHA, RAW_SHA], settings=settings,
            rgb={**{name: hash_pixels(frame) for name, frame in trial.items()}, 'raw_on': hash_pixels(raw_on)},
            rows=rows, gradient_correlation=aligned, passed_distance=pass_distance, passed_clipping=pass_clip,
            passed=pass_distance and pass_clip and aligned >= .95, files=files, panel_sha=board))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('index', type=int, choices=(0, 1))
    run(**vars(parser.parse_args()))
