"""Score existing street video exports on aligned independent samples."""

import argparse
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_spatial_real_occlusion import FIXTURES, source_frames
from probe_spatial_video_history import ffmpeg
from probe_spatial_video_stage_timing import checked


PARENT_SHA = '7af5208010d95fcac70a17170feaf3f8d39f6a45b64ccfb7acdec6d5fc008ec6'
NATIVE_SHA = '026fc4a382e7a3897753f8e1263f2602f352864d274a95ff3c79fc5fcf4425bd'
LUMA = np.array([.2126, .7152, .0722])


def samples(source, original):
    with original.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != NATIVE_SHA:
            raise ValueError('Changed native frame archive')
    fixture = FIXTURES['street']
    checked(source.parent, source.name, fixture.sha256)
    target = source_frames(source, False, fixture)
    raw = ffmpeg('-i', original, '-vf', 'scale=640:360', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
    frames = np.frombuffer(raw, np.uint8).reshape(-1, *fixture.shape)
    digests = {sha256(frame.tobytes()).digest(): i for i, frame in enumerate(frames)}
    if len(frames) != len(digests) or len(frames) != 72:
        raise ValueError('Expected 72 distinct native frames')
    mapping = [digests[sha256(frame.tobytes()).digest()] for frame in target]
    if sorted(set(mapping)) != list(range(72)) or mapping != sorted(mapping):
        raise ValueError('Changed frame mapping')
    return target, mapping


def resample(image, yy, xx, shift):
    y, x = (yy + shift[0], xx + shift[1])
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    dy, dx = y - y0, x - x0
    if image.ndim == 3:
        dy, dx = dy[..., None], dx[..., None]
    return ((1-dy)*((1-dx)*image[y0,x0] + dx*image[y0,x0+1])
            + dy*((1-dx)*image[y0+1,x0] + dx*image[y0+1,x0+1]))


def align(left, right, area):
    y, x = area
    yy, xx = np.mgrid[y.start-6:y.stop+6, x.start-6:x.stop+6]
    ring = (yy < y.start) | (yy >= y.stop) | (xx < x.start) | (xx >= x.stop)
    yy, xx = yy[ring], xx[ring]
    target = left[yy, xx] @ LUMA
    target -= target.mean()

    def correlation(shift):
        warped = resample(right, yy, xx, shift) @ LUMA
        warped -= warped.mean()
        norm = np.linalg.norm(target) * np.linalg.norm(warped)
        return float(target @ warped / norm) if norm else -1.

    choices = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
    winner = max(choices, key=correlation)
    refined = [(float(winner[0]+dy), float(winner[1]+dx)) for dy in np.arange(-.75, 1, .25)
               for dx in np.arange(-.75, 1, .25) if abs(winner[0]+dy) <= 2 and abs(winner[1]+dx) <= 2]
    best = max(refined, key=correlation)
    return best, correlation(best)


def score_pair(source, actual, a, b, area):
    left, right = source[a], source[b]
    shift, correlation = align(left, right, area)
    y, x = area
    yy, xx = np.mgrid[y.start:y.stop, x.start:x.stop]
    before = left[area]
    after = resample(right, yy, xx, shift)
    input_mae = float(np.abs(before-after).mean())
    values = None
    if correlation >= .98 and input_mae <= 1:
        values = {label: float(np.abs((resample(rows[b], yy, xx, shift)-after)
                                     - (rows[a][area].astype(float)-before)).mean()) for label, rows in actual.items()}
    return {'indices': [a, b], 'correlation': correlation, 'shift': shift, 'input_mae': input_mae,
            'unaligned_input_mae': float(np.abs(left[area].astype(float)-right[area]).mean()), 'score': values}


def load_exports(parent, ledger):
    outputs = {}
    fixture = FIXTURES['street']
    for arm, row in ledger['results'].items():
        folder = parent / arm
        frames = {}
        for index, digest in row['export']['pngs'].items():
            raw = checked(folder, f'{int(index):03d}.png', digest)
            with Image.open(BytesIO(raw)) as image:
                frames[int(index)] = np.array(image)
        md5 = ffmpeg('-i', folder / 'result.mp4', '-map', '0:v:0', '-pix_fmt', 'rgb24', '-fps_mode', 'passthrough',
                     '-f', 'framemd5', 'pipe:1')
        if sha256(md5).hexdigest() != row['export']['md5_sha256']:
            raise ValueError('Changed original video export')
        outputs[arm + '_rgb'] = frames
        outputs[arm + '_mp4'] = source_frames(folder / 'result.mp4', False, fixture)
    return outputs


def run(source, original, parent, out):
    ledger = json.loads(checked(parent, 'results.json', PARENT_SHA))
    video, mapping = samples(source, original)
    exports = load_exports(parent, ledger)
    fixture = FIXTURES['street']
    results = {}
    for name, (start, stop) in fixture.windows.items():
        unique = [i for i in range(start, stop) if i == start or mapping[i] != mapping[i-1]]
        pairs = [(a, b) for a, b in zip(unique, unique[1:]) if mapping[b] == mapping[a] + 1]
        rows = {}
        for label, area in fixture.areas.items():
            log = [score_pair(video, exports, a, b, area) for a, b in pairs]
            keys = [record['score'] for record in log if record['score'] is not None]
            rows[label] = {'native_pairs': len(log), 'raw_eligible': sum(x['unaligned_input_mae'] <= 1 for x in log),
                           'aligned_eligible': len(keys), 'max_drift': {arm: max(x[arm] for x in keys) for arm in exports}
                           if keys else None, 'pairs': log}
        results[name] = rows
    out.mkdir()
    (out / 'results.json').write_text(json.dumps({'native_map': mapping, 'windows': results}, indent=2) + '\n',
                                      encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'original', 'parent', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
