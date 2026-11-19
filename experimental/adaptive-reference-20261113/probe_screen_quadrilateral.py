"""Check whether visible closed quadrilaterals can veto screen skin by geometry."""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from probe_reference_skin_chroma_attribution import regions
from probe_reference_skin_chroma_isolation import BG_SHA, checked_json, checked_rgb, json_to
from probe_screen_face_transfer import inputs
from probe_skin_screen_context import BASE, NEGATIVE, digest, sources


def screen_geometry(rgb, cv):
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(cv.GaussianBlur(gray, (3, 3), 0), 40, 100)
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    covered = np.zeros(gray.shape, np.uint8)
    accepted = []
    for contour in contours:
        quad = cv.approxPolyDP(contour, .02 * cv.arcLength(contour, True), True)
        if len(quad) != 4 or not cv.isContourConvex(quad):
            continue
        if not .015 * gray.size <= cv.contourArea(quad) <= .8 * gray.size:
            continue
        corners = quad[:, 0]
        sides = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(sides, axis=1)
        if lengths.min() < .08 * min(gray.shape):
            continue
        angle_cos = (sides * np.roll(sides, 1, axis=0)).sum(axis=1) / (lengths * np.roll(lengths, 1))
        if np.any(np.abs(angle_cos) > .5):
            continue
        cv.fillConvexPoly(covered, quad, 1)
        accepted.append(corners.tolist())
    return covered.astype(bool), accepted


def cases():
    old = BASE / 'skin-screen-boundary'
    anchor = BASE / 'bg-real-editor-look'
    receipt = checked_json(anchor / 'results.json', BG_SHA)
    for pair, rgb in sources():
        item = receipt['cases'][pair]['artifacts']
        np.testing.assert_array_equal(rgb, checked_rgb(anchor / pair / 'target.png', item['target.png']))
        yield pair, rgb, old / f'{pair}-skin.npy', regions()[pair]
    synthetic = np.asarray(Image.open(old / 'composite-input.png').convert('RGB'))
    yield 'foreground', synthetic, old / 'foreground-skin.npy', {'foreground forehead': np.s_[75:81, 125:129]}
    for key, rgb, (l, t, r, b), screen in inputs():
        name = 'screen face' if screen else 'direct face'
        yield key, rgb, BASE / 'screen-face-transfer' / f'{key}-weight.npy', {name: np.s_[t:b, l:r]}


def render(rgb, weight):
    return np.rint(np.clip(rgb + weight[..., None] * [-8, 0, 12], 0, 255)).astype(np.uint8)


def assess(rgb, old, new, veto, area, negative):
    touch = float((new[area] > .05).mean())
    original_touch = float((old[area] > .05).mean())
    active = float((new[area] > .5).mean())
    original_active = float((old[area] > .5).mean())
    if negative:
        passed = touch <= .01
    elif original_active >= .8:
        passed = active >= .8
    else:
        passed = original_active - active <= .05
    return dict(touch=touch, original_touch=original_touch, active=active, original_active=original_active, passed=passed,
                covered=float(veto[area].mean()), source_pixels=int(rgb[area].shape[0] * rgb[area].shape[1]))


def run(out, cv):
    out.mkdir()
    rows = {}
    for name, rgb, path, boxes in cases():
        old = np.load(path, allow_pickle=False)
        if old.shape != rgb.shape[:2] or not np.isfinite(old).all():
            raise ValueError(f'Invalid frozen mask for {name}')
        veto, accepted = screen_geometry(rgb, cv)
        new = old * ~veto
        frames = np.concatenate((rgb, render(rgb, old), render(rgb, new)))
        target = out / f'{name}.png'
        Image.fromarray(frames).save(target)
        results = {label: assess(rgb, old, new, veto, area, label in NEGATIVE or label == 'screen face')
                   for label, area in boxes.items()}
        rows[name] = dict(areas=results, accepted_quads=accepted, panel_sha256=digest(target), mask_sha256=digest(path))
    if len(rows) != 10:
        raise ValueError('Ten locked images expected')
    json_to(out / 'results.json', dict(cases=rows, passed=all(row['passed'] for item in rows.values() for row in item['areas'].values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('locator_dependencies', type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    import cv2
    run(args.out, cv2)
