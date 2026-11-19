"""Replay a photographed edit only within verified same-photo overlap."""

import argparse
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from background_intervention import target_regions
from probe_playraw_mixed_light_appearance import STORE, STORE_SHA, image_measure
from probe_playraw_mixed_light_transfer import BASE, LUMA, PARENT, PARENT_SHA, STRENGTH, WHITE, chroma, read, regions
from probe_sparks_shared_chroma import render
from probe_synclight_cross_view import accepted_points
from probe_target_mirror_chroma import checked_json, checked_rgb, json_to, save
from spatial_render import finish_spatial
from verify_development_inputs import manifest


def warp(image, matrix, target):
    return cv2.warpAffine(image, matrix, (target.shape[1], target.shape[0]), flags=cv2.INTER_LINEAR)


def texture_similarity(target, guide, support):
    kernel = np.ones((3, 3), np.uint8)
    area = cv2.erode(support.astype(np.uint8), kernel).astype(bool)[::2, ::2]
    readings = []
    for image in (target, guide):
        light = (image @ LUMA).astype(np.float32)
        readings.append(np.concatenate([cv2.Sobel(light, cv2.CV_32F, dx, dy)[::2, ::2][area]
                                        for dx, dy in ((1, 0), (0, 1))]))
    a, b = readings
    if min(a.std(), b.std()) == 0:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def align(sift, target, reference):
    attempts = []
    for mirrored, guide in ((False, reference), (True, reference[:, ::-1])):
        status, pts = accepted_points(sift, target, guide)
        attempts.append({**status, 'mirrored': mirrored, 'accepted': False})
        if pts is None:
            continue
        matrix, inliers = cv2.estimateAffinePartial2D(pts[1], pts[0], method=cv2.RANSAC,
                                                      ransacReprojThreshold=1.)
        if matrix is None or int(inliers.sum()) < 12:
            continue
        if (np.abs(matrix[:, :2] - np.eye(2)).max() <= .002
                and np.abs(matrix[:, 2] - np.rint(matrix[:, 2])).max() <= .125):
            matrix[:, :2] = np.eye(2)
            matrix[:, 2] = np.rint(matrix[:, 2])
        support = warp(np.full(guide.shape[:2], 255, np.uint8), matrix, target) == 255
        coverage = float(support.mean())
        if coverage < .55:
            continue
        placed = warp(guide, matrix, target)
        similarity = texture_similarity(target, placed, support)
        attempts[-1].update(overlap=coverage, gradient_correlation=similarity, matrix=matrix.tolist())
        if similarity >= .98:
            attempts[-1]['accepted'] = True
            return placed, support, attempts
    return None, None, attempts


def transfer(sift, target, reference, baseline):
    aligned, support, attempts = align(sift, target, reference)
    if aligned is None:
        return baseline.copy(), attempts
    result = target.copy()
    target_color = chroma(target[support])
    wanted = target_color + STRENGTH * (chroma(aligned[support]) - target_color)
    result[support], _ = render(wanted, target[support] @ LUMA)
    np.testing.assert_array_equal(result[~support], target[~support])
    return result, attempts


def run(out):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create()
    prior = checked_json(STORE / 'results.json', STORE_SHA)
    provenance = checked_json(PARENT / 'results.json', PARENT_SHA)
    original, _ = read(PARENT / 'off.tif', False)
    edited, mask = read(PARENT / 'on.tif', True)
    if ([sha256(x.tobytes()).hexdigest() for x in (original, edited)] != provenance['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != provenance['mask_pixels_sha256']):
        raise ValueError('Changed author images')
    expected = chroma(finish_spatial(original, edited, STRENGTH)) - chroma(original)
    neutral = target_regions(original, np.zeros_like(mask, bool))['neutral_proxy']
    zones = {'author': mask >= 204, 'protected': mask == 0, 'neutral': neutral & (mask == 0)}
    h, w = mask.shape
    crop = np.s_[h//16:h-h//16, w//16:w-w//16]
    pairs = {'full': (original, edited), 'mirror': (original[:, ::-1], edited[:, ::-1]),
             'crop': (original[crop], edited[crop]), 'white': (original, edited)}
    bg = manifest(BASE, 'bg-real-editor-look')['cases']
    white = checked_rgb(BASE / 'bg-real-editor-look' / WHITE / 'target.png', bg[WHITE]['artifacts']['target.png'])
    jersey = np.zeros(white.shape[:2], bool)
    jersey[regions()[WHITE]['白球服']] = True
    out.mkdir()
    cases = {}
    for key, pair in pairs.items():
        cross = key == 'white'
        reference = prior['white'] if cross else prior['cases'][key]
        frames, checks = [], []
        for state, guide in zip(('off', 'on'), pair, strict=True):
            base = checked_rgb(STORE / f'{key}-{state}.png', reference['outputs'][state]['image_sha256'])
            frame, attempts = transfer(sift, white if cross else original, guide, base)
            checks.append({'attempts': attempts, 'accepted': any(x['accepted'] for x in attempts),
                           'fallback_exact': bool(np.array_equal(frame, base)),
                           'hash': save(out / f'{key}-{state}.png', frame)})
            frames.append(frame)
        result = image_measure(frames, None if cross else expected, {'white': jersey} if cross else zones, cross)
        proper = all(row['accepted'] != cross for row in checks)
        exact = all(row['fallback_exact'] for row in checks) if cross else np.array_equal(frames[0], original)
        cases[key] = {'analysis': checks, 'metrics': result,
                      'passed': bool(result['passed'] and proper and exact)}
    negatives = {}
    for key, case in bg.items():
        anchor = BASE / 'bg-real-editor-look' / key
        paths = case['artifacts']
        original_bg = checked_rgb(anchor / 'target.png', paths['target.png'])
        for role in ('C1', 'F1'):
            name = f'{role}-off-reference.png'
            guide = checked_rgb(anchor / name, paths[name])
            negatives[f'{key}-{role}'] = align(sift, original_bg, guide)[2]
    rejected = len(negatives) == 8 and all(not any(row['accepted'] for row in x) for x in negatives.values())
    json_to(out / 'results.json', {'cases': cases, 'negatives': negatives,
                                   'passed': rejected and all(row['passed'] for row in cases.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
