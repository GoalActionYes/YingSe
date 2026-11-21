"""Render a fixed grade from verified binocular correspondences on real photos."""

from argparse import ArgumentParser
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
import torch

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, quantize
from probe_gros_film_b_cross import FILM
from probe_synclight_surface_anchors import BASE, STRENGTH, fitted, score
from probe_target_mirror_chroma import CHECKPOINT, digest, save
from spatial_render import SpatialEngine, finish_spatial


SCENES = {
    'Art': '087f18c9c27d0a4facd5e07df89ab7f04a4653207d7e8ad6dc08fc138160c15f',
    'Dolls': 'fdfc295f9270e1a26d5ec6705030107541178d1b72bb699909044c27367500b1',
}
LUMA = np.array([.2126, .7152, .0722])


def scene_from_zip(source, scene, expected):
    path = source / f'{scene}-2views.zip'
    if digest(path) != expected:
        raise ValueError(f'Unexpected archive identity: {scene}')
    with ZipFile(path) as archive:
        frames = []
        for stem, mode in [('view1', 'RGB'), ('view5', 'RGB'), ('disp1', 'L'), ('disp5', 'L')]:
            with Image.open(BytesIO(archive.read(f'{scene}/{stem}.png'))) as image:
                frames.append(np.asarray(image.convert(mode)))
    if len({frame.shape[:2] for frame in frames}) != 1:
        raise ValueError(f'Inconsistent raster sizes: {scene}')
    return frames


def consistent_surface(disparity):
    kernel = np.ones((3, 3), np.uint8)
    low = cv2.erode(disparity, kernel)
    high = cv2.dilate(disparity, kernel)
    return (low > 0) & (high - low <= 1)


def stereo_maps(left, right):
    y, x = np.indices(left.shape, dtype=np.float32)
    shift_x = x - left
    def warp(frame, mode):
        return cv2.remap(frame, shift_x, y, mode)
    reverse = warp(right, cv2.INTER_NEAREST)
    right_safe = warp(consistent_surface(right).astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
    keep = ((shift_x >= 0) & (shift_x < left.shape[1] - 1)
            & consistent_surface(left) & right_safe & (np.abs(left - reverse) <= 1))
    return keep, warp


def display_style(frame, stages):
    view, inverse, film = stages
    codes = frame.astype(np.float32) / 255
    return quantize(np.clip(converted(converted(converted(codes, inverse), film), view), 0, 1))


def mean_error(a, b, mask):
    return float(np.abs(a[mask].astype(np.float64) - b[mask]).mean())


@torch.inference_mode()
def run(source, out):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cube = BASE / 'arri-second-film-assets-v1/second-film.cube'
    if digest(cube) != FILM:
        raise ValueError('Changed Film B')
    stages = transforms(cube)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 baseline')
    out.mkdir()
    report = {'archives': SCENES, 'film_sha256': FILM, 'checkpoint_sha256': CHECKPOINT, 'records': {}}
    for scene, expected in SCENES.items():
        target, reference, left, right = scene_from_zip(source, scene, expected)
        valid, warp = stereo_maps(left.astype(np.float32) / 3, right.astype(np.float32) / 3)
        if valid.sum() < 5000 or valid.mean() < .25:
            raise ValueError(f'Insufficient valid stereo coverage: {scene}')
        y, x = np.indices(valid.shape)
        fit = valid & (((x // 32 + y // 32) % 2) == 0)
        hold = valid & ~fit
        edited = display_style(reference, stages)
        left_samples = target[fit].astype(np.float64)
        right_fit = reference[fit].astype(np.float64)
        warped_original = warp(reference.astype(np.float32), cv2.INTER_LINEAR)
        warped_edited = warp(edited.astype(np.float32), cv2.INTER_LINEAR)
        arms = {
            'disparity': (left_samples, (warped_original[fit], warped_edited[fit])),
            'same_view': (right_fit, (right_fit, edited[fit])),
            'same_coordinates': (left_samples, (right_fit, edited[fit])),
        }
        outputs, coefficients = {}, {}
        for arm, (samples, states) in arms.items():
            grades, fits = zip(*(fitted(target, samples, state.astype(np.float64))
                                 for state in states), strict=True)
            outputs[arm] = grades
            coefficients[arm] = fits
        outputs['baseline_13'] = tuple(
            finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
            for ref in (reference, edited)
        )
        answer = finish_spatial(target, display_style(target, stages), STRENGTH)
        y_code = target @ LUMA
        neutral = (np.ptp(target, axis=2) <= 20) & (y_code >= 40) & (y_code <= 220)
        masks = {'whole': np.ones_like(valid), 'held_correspondence': hold, 'neutral': neutral}
        masks = {key: mask for key, mask in masks.items() if mask.sum() >= 256}
        folder = out / scene
        folder.mkdir()
        hashes = {'authored.png': save(folder / 'authored.png', answer)}
        scores = {}
        for arm, grades in outputs.items():
            scores[arm] = score(target, answer, grades, masks)
            for state, grade in zip(('off', 'on'), grades, strict=True):
                name = f'{arm}-{state}.png'
                hashes[name] = save(folder / name, grade)
        report['records'][scene] = {
            'coverage': float(valid.mean()), 'fit_pixels': int(fit.sum()),
            'region_pixels': {key: int(mask.sum()) for key, mask in masks.items()},
            'native_held_mae': mean_error(target, reference, hold),
            'warped_held_mae': mean_error(target, warped_original, hold),
            'scores': scores, 'coefficients': coefficients, 'image_sha256': hashes,
        }

    (out / 'results.json').write_bytes((json.dumps(report, indent=2, allow_nan=False) + '\n').encode())


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
