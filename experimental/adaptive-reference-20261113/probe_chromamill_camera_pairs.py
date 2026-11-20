"""Acquire the fixed camera pairs and verify geometric and matched-edit support."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from probe_beyond_rgb_transfer import digest, save, write_json
from probe_chromamill_capture_cross import REPO, REVISION, decode_rgb8
from probe_inretouch_registration import gradient, shifts
from probe_physical_blue_global_fit import display, linear


PLAN = '98582654231b142ecf3a0bf6dd29768c261ad56e1d0456eb925c9db29ddef629'
SCENES = ('Scene3_a', 'Scene9_b', 'Scene10_a')
CAMERAS = ('nikon', 'phone')
LIGHTS = {'white': '3127-3290', 'green': '2821-4727'}


def planned_assets(plan):
    if digest(plan) != PLAN:
        raise ValueError('Unexpected fixed source plan')
    rows = json.loads(plan.read_bytes())
    if rows['revision'] != REVISION:
        raise ValueError('Unexpected CHROMA-MILL revision')
    assets = {f'{camera}-{scene}-{name}': rows['selected_train'][f'{camera}/{scene}/{xy}']
              for camera, scene, (name, xy) in product(CAMERAS, SCENES, LIGHTS.items())}
    return assets


def verified(path, row):
    if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
        raise ValueError(f'Changed or partial published file: {path.name}')
    return path


def retrieve(key, row, cache, old, proxy):
    if key.startswith('phone-') and key.endswith('-white'):
        name = key.removeprefix('phone-')
        return verified(old / f'{name}-241.png', row)
    path = cache / f'{key}.png'
    if not path.is_file():
        scratch = path.with_suffix('.partial')
        source = f"{REPO}/resolve/{REVISION}/{row['path']}"
        subprocess.run(['curl.exe', '--proxy', proxy, '--fail', '--silent', '--show-error',
                        '--location', '--proto', '=https', '--proto-redir', '=https',
                        '--max-time', '100', '--max-filesize', str(row['bytes']),
                        '--output', str(scratch), source], check=True)
        verified(scratch, row).replace(path)
        print(f"{key}: {row['bytes']:,} bytes verified", flush=True)
    return verified(path, row)


def evaluate(white, green):
    if white.shape != green.shape:
        raise ValueError('Differing prepared geometry within one camera')
    reference, capture = (linear(rgb) for rgb in (white, green))
    first, second = (frame.mean(axis=(0, 1)) for frame in (reference, capture))
    if min(first.min(), second.min()) <= 0:
        raise ValueError('No support for per-channel color matching')
    gain = second / first
    tinted = reference * gain
    clipped = float((tinted.max(axis=2) > 1).mean())
    edit = display(tinted)
    offset = float(np.abs(linear(edit).mean(axis=(0, 1)) - second).max())
    change = float(np.linalg.norm(np.log(gain[[0, 2]] / gain[1])))
    registered = shifts(gradient(white), gradient(green))[0]
    qualified = registered['zero'] >= .60 and registered['peak_gain'] <= .02
    qualified = qualified and change >= .08 and offset <= .02 and clipped <= .05
    return dict(registered=registered, linear_global_gain=gain.tolist(), cast_distance=change,
                digital_mean_error=offset, digital_clipped_fraction=clipped, qualified=qualified), edit


def run(plan, old, cache, out, proxy):
    assets = planned_assets(plan)
    download_size = sum(row['bytes'] for key, row in assets.items()
                        if not (key.startswith('phone-') and key.endswith('-white')))
    if download_size != 139_294_691 or download_size > 192_000_000:
        raise ValueError('Changed retrieval budget')
    cache.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as worker:
        futures = {key: worker.submit(retrieve, key, row, cache, old, proxy) for key, row in assets.items()}
        acquired = {key: future.result() for key, future in futures.items()}

    out.mkdir()
    measured = {}
    for scene in SCENES:
        board = Image.new('RGB', (990, 480), '#242630')
        draw = ImageDraw.Draw(board)
        measured[scene] = {}
        for index, camera in enumerate(CAMERAS):
            key = f'{camera}-{scene}'
            white, green = [decode_rgb8(acquired[f'{key}-{light}']) for light in LIGHTS]
            scores, edited = evaluate(white[0], green[0])
            outputs = {'white': white[0], 'green': green[0], 'digital': edited}
            scores['prepared_sha256'] = {label: save(out / f'{key}-{label}.png', image)
                                         for label, image in outputs.items()}
            scores['source_formats'] = {'white': white[1], 'green': green[1]}
            measured[scene][camera] = scores
            for column, (label, rgb) in enumerate(outputs.items()):
                x, y = 5 + column * 330, index * 240
                thumbnail = Image.fromarray(rgb)
                thumbnail.thumbnail((320, 213), Image.Resampling.LANCZOS)
                draw.text((x, y + 4), f'{camera} / {label}', fill='white')
                board.paste(thumbnail, (x, y + 22))
        path = out / f'{scene}-comparison.png'
        board.save(path)
        measured[scene]['panel_sha256'] = digest(path)
    qualified = [scene for scene in SCENES if all(measured[scene][camera]['qualified'] for camera in CAMERAS)]
    write_json(out/'report.json', dict(plan_sha256=PLAN, assets=assets, new_bytes=download_size,
                                      qualified_scenes=qualified, admission_pass=len(qualified) >= 2, scenes=measured))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('plan', 'old', 'cache', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--opencv-site', type=Path)
    args = vars(parser.parse_args())
    cv_site = args.pop('opencv_site')
    if cv_site is not None:
        sys.path.append(str(cv_site.resolve()))
    run(**args)
