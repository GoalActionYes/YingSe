"""Prepare fixed Sparks archival pairs and compare the existing two-image grade."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image


FRAMES = (4592, 9184)
FRAME_BYTES = 53_119_100
SIZE = (512, 270)
AP0 = ((.7347, .2653), (0., 1.), (.0001, -.077))
REC709 = ((.64, .33), (.30, .60), (.15, .06))
D60 = (.32168, .33767)
D65 = (.3127, .3290)
BRADFORD = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367],
                     [.0389, -.0685, 1.0296]])
PROTOCOL = Path(__file__).with_name('sparks-archival-grade-protocol.md')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, record):
    path.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode())


def xyz(xy):
    x, y = xy
    return np.array([x / y, 1., (1 - x - y) / y])


def rgb_to_xyz(primaries, white):
    primary_matrix = np.column_stack([xyz(p) for p in primaries])
    return primary_matrix * np.linalg.solve(primary_matrix, xyz(white))


def display_transform():
    display = rgb_to_xyz(REC709, D65)
    adaptation = np.linalg.solve(BRADFORD, (BRADFORD @ xyz(D65) / (BRADFORD @ xyz(D60)))[:, None] * BRADFORD)
    transform = np.linalg.solve(display, adaptation @ rgb_to_xyz(AP0, D60))
    np.testing.assert_allclose(transform @ np.ones(3), 1., atol=1e-12)
    return transform, display[1]


def prepare_thumbnail(source, matrix, luma):
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2

    bgr = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (2160, 4096, 3) or bgr.dtype != np.float32:
        raise ValueError(f'Unexpected EXR format for {source}')
    if not np.isfinite(bgr).all():
        raise ValueError(f'EXR contains non-finite values: {source}')
    small = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float64)
    linear = small @ matrix.T
    luminance = linear @ luma
    mapped = linear / (1 + np.maximum(luminance, 0.)[..., None])
    clipped = np.clip(mapped, 0., 1.)
    encoded = np.where(clipped <= .0031308, clipped * 12.92, 1.055 * clipped**(1 / 2.4) - .055)
    rgb = np.rint(encoded * 255.).astype(np.uint8)
    return rgb, dict(ap0_min=float(bgr.min()), ap0_max=float(bgr.max()),
                     below_zero=float((mapped < 0.).mean()), above_one=float((mapped > 1.).mean()))


def grad(rgb, luma):
    plane = rgb @ luma
    dy = plane[2:, 1:-1] - plane[:-2, 1:-1]
    dx = plane[1:-1, 2:] - plane[1:-1, :-2]
    return np.hypot(dx, dy).ravel()


def prepare(source, out):
    started = time.monotonic()
    intake = source / 'intake.json'
    files = {item['name']: item for item in json.loads(intake.read_text(encoding='utf-8-sig'))}
    expected = {f'{role}-{frame}' for frame in FRAMES for role in ('nam', 'gam')}
    if set(files) != expected or sum(r['size'] for r in files.values()) > 204 * 1024**2:
        raise ValueError('Unexpected frame set or size')
    matrix, luma = display_transform()
    out.mkdir()
    pairs = {}
    for frame in FRAMES:
        thumbs = {}
        rows = {}
        for role in ('nam', 'gam'):
            name = f'{role}-{frame}'
            path = source / f'{name}.exr'
            if path.stat().st_size != FRAME_BYTES or sha(path) != files[name]['sha256']:
                raise ValueError(f'Invalid frame identity: {name}')
            thumb, info = prepare_thumbnail(path, matrix, luma)
            png = out / f'{name}.png'
            Image.fromarray(thumb).save(png)
            rows[role] = dict(files[name], **info, png_sha256=sha(png))
            thumbs[role] = thumb
        nam, gam = thumbs['nam'], thumbs['gam']
        delta = float(np.abs(nam.astype(np.int16) - gam).mean())
        gradients = [grad(rgb, luma) for rgb in (nam, gam)]
        corr = float(np.corrcoef(gradients)[0, 1]) if all(g.std() > 0 for g in gradients) else None
        pairs[frame] = dict(frames=rows, gradient_correlation=corr, image_delta=delta,
                            qualified=corr is not None and corr >= .70 and delta >= 2.)
    if time.monotonic() - started > 120:
        raise RuntimeError('Preparation exceeded the compute budget')
    write_json(out / 'prepared.json', dict(pairs=pairs, source_sha256=sha(Path(__file__)),
                                           protocol_sha256=sha(PROTOCOL), intake_sha256=sha(intake)))
    print({frame: (r['gradient_correlation'], r['image_delta'], r['qualified']) for frame, r in pairs.items()})


def photo(folder, role, frame, prepared):
    path = folder / f'{role}-{frame}.png'
    if sha(path) != prepared['pairs'][str(frame)]['frames'][role]['png_sha256']:
        raise ValueError(f'Prepared PNG changed: {path}')
    with Image.open(path) as image:
        return np.array(image)


def render(source, out):
    import torch

    from spatial_render import SpatialEngine, finish_spatial
    from independent_coco import ROOT

    started = time.monotonic()
    torch.set_num_threads(4)
    manifest = source / 'prepared.json'
    prepared = json.loads(manifest.read_bytes())
    if not all(row['qualified'] for row in prepared['pairs'].values()):
        raise ValueError('Both archival pairs must qualify')
    engine = SpatialEngine('multiref-v1')
    reference = ROOT / 'docs/research/2026-11-18-synclight-real-pixel-evidence.json'
    if engine.checkpoint_sha != json.loads(reference.read_bytes())['weight_sha256']:
        raise ValueError('Expected the same frozen 13 checkpoint')
    out.mkdir()
    scores = {}
    for frame, donor in (FRAMES, FRAMES[::-1]):
        target = photo(source, 'nam', frame, prepared)
        guide = photo(source, 'gam', donor, prepared)
        result = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
        path = out / f'grade-{frame}-from-{donor}.png'
        Image.fromarray(result).save(path)
        # The same-content author grade is used only after rendering the output.
        expected = photo(source, 'gam', frame, prepared).astype(np.int16)
        neutral = (np.ptp(target, axis=-1) <= 12) & (target.min(-1) >= 32) & (target.max(-1) <= 240)
        before = np.abs(target - expected)
        after = np.abs(result - expected)
        scores[frame] = dict(donor=donor, neutral_pixels=int(neutral.sum()),
                             whole_mae=[float(before.mean()), float(after.mean())],
                             neutral_mae=[float(before[neutral].mean()), float(after[neutral].mean())]
                             if neutral.sum() >= 256 else None, output_sha256=sha(path))
    if time.monotonic() - started > 120:
        raise RuntimeError('Comparison exceeded the compute budget')
    write_json(out / 'results.json', dict(scores=scores, prepared_sha256=sha(manifest),
                                          weights_sha256=engine.checkpoint_sha, strength=.8))
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'render'))
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    {'prepare': prepare, 'render': render}[args.phase](args.source, args.out)
