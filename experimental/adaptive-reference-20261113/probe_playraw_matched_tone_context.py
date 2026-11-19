"""Replay a known editor grade across two tonal contexts with matching off RGB."""

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from probe_hdm_hdr_display import json_to
from probe_playraw_mixed_light_cross_editor import INPUTS, default_states
from probe_playraw_mixed_light_skin import export, read
from probe_playraw_mixed_light_transfer import LUMA, STRENGTH, chroma, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import save
from spatial_render import finish_spatial


COLORS = np.array([[80, 51, 42], [102, 70, 56], [120, 84, 72], [144, 104, 87],
                   [162, 115, 95], [183, 135, 112], [201, 148, 128], [219, 171, 150],
                   [98, 68, 50], [134, 86, 73], [172, 122, 99], [211, 161, 130],
                   [60, 60, 60], [99, 99, 99], [144, 144, 144], [192, 192, 192]], dtype=np.uint8)
EXPONENT = .55


def curve(rgb, exponent):
    return np.rint(255 * (rgb.astype(np.float64) / 255) ** exponent).astype(np.uint8)


def chart():
    return COLORS.reshape(4, 4, 3).repeat(64, axis=0).repeat(256, axis=1)


def patches():
    for i in range(len(COLORS)):
        y, x = divmod(i, 4)
        yield np.s_[y * 64 + 24:y * 64 + 40, x * 256 + 120:x * 256 + 136]


def score(a, b, out):
    off_a, on_a = a
    off_b, on_b = b
    answer_a = finish_spatial(off_a, on_a, STRENGTH)
    answer_b = finish_spatial(off_b, on_b, STRENGTH)
    move = chroma(answer_a) - chroma(off_a)
    control, _ = render(chroma(off_a) + move, off_a @ LUMA)
    replay, _ = render(chroma(off_b) + move, off_b @ LUMA)
    save(out / 'a-expected.png', answer_a)
    save(out / 'b-expected.png', answer_b)
    images = {'a-off': save(out / 'a-off.png', off_a), 'b-off': save(out / 'b-off.png', off_b),
              'b-replay': save(out / 'b-replay.png', replay)}
    rows = []
    for i, area in enumerate(patches()):
        starter = [img[area] for img in (off_a, on_a, off_b, on_b)]
        off_max = int(np.abs(starter[0].astype(np.int16) - starter[2]).max())
        unclipped = all(np.min(tile) > 0 and np.max(tile) < 255 for tile in starter)
        wanted = (chroma(answer_b[area]) - chroma(off_b[area])).reshape(-1, 2)
        actual = (chroma(replay[area]) - chroma(off_b[area])).reshape(-1, 2)
        response = float(np.median(np.abs(wanted).mean(axis=1)))
        metric = stats(actual, wanted, np.ones(len(wanted), bool))
        control_mae = float(np.abs(chroma(control[area]) - chroma(answer_a[area])).mean())
        eligible = off_max <= 1 and unclipped and response >= 2
        rows.append({'rgb': COLORS[i].tolist(), 'skin': i < 12, 'off_max': off_max, 'unclipped': unclipped,
                     'expected_median': response, 'control_mae': control_mae, 'cross': metric,
                     'eligible': eligible, 'different': eligible and metric['response_mae'] > 3})
    skin = [row for row in rows if row['skin'] and row['eligible']]
    return {'patches': rows, 'eligible_skin': len(skin), 'different_skin': sum(row['different'] for row in skin),
            'images': images, 'qualified': len(skin) >= 8 and all(row['control_mae'] <= 1 for row in skin)}


def run(root, out):
    if not all(path.is_absolute() and path.as_posix().isascii() for path in (root, out)):
        raise ValueError('Use absolute ASCII paths')
    paths = {key: root / INPUTS[key][0] for key in ('author', 'cli')}
    for key, path in paths.items():
        if sha256(path.read_bytes()).hexdigest() != INPUTS[key][1]:
            raise ValueError(f'Changed {key}')
    settings = default_states(paths['author'].read_bytes())
    photo = chart()
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    editor_env = {key: value for key, value in os.environ.items() if key != 'OMP_NUM_THREADS'}
    out.mkdir()
    trials = []
    for i in range(2):
        folder = out / str(i)
        folder.mkdir()
        runs = []
        for label, pixels in [('a', photo), ('b', curve(photo, EXPONENT))]:
            path = folder / f'{label}.tif'
            Image.fromarray(pixels).save(path, icc_profile=icc)
            pair = []
            for state, xmp in zip(('off', 'on'), settings, strict=True):
                name = f'{label}-{state}'
                sidecar = folder / f'{name}.xmp'
                sidecar.write_bytes(xmp)
                frame, _ = read(export(paths['cli'], path, sidecar, folder, name, False, env=editor_env), False)
                if frame.shape != photo.shape:
                    raise ValueError('Changed chart dimensions')
                pair.append(frame if label == 'a' else curve(frame, 1 / EXPONENT))
            runs.append(pair)
        trials.append(score(*runs, folder))
    repeatable = [[(p['eligible'], p['different']) for p in row['patches']] for row in trials]
    json_to(out / 'results.json', {'trials': trials, 'same_classification': repeatable[0] == repeatable[1],
                                  'demonstrated': repeatable[0] == repeatable[1] and all(
                                      row['qualified'] and row['different_skin'] >= 4 for row in trials)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
