"""Bound the coverage achievable by any filter on the prior mutual SIFT matches."""

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np

from probe_synclight_cross_view import BASE, PIN, ROOT, detect, digest, load


LAMP_REPORT = '2f9e6cda822c3811e087548a636f98133523e1bc3c280df5fbf9df70e2bbc7a5'


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Unexpected manifest identity: {path}')
    return json.loads(path.read_bytes())


def coverage(points, image_shape):
    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(points)) / np.prod(image_shape[:2]))


def mutual_matches(first, second):
    first_keypoints, first_descriptors = first
    second_keypoints, second_descriptors = second
    if min(len(first_keypoints), len(second_keypoints)) < 2:
        return np.empty((2, 0, 2), dtype=np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_matches(query, train):
        return {
            best.queryIdx: best.trainIdx
            for best, runner_up in matcher.knnMatch(query, train, k=2)
            if best.distance < .75 * runner_up.distance
        }

    forward = ratio_matches(first_descriptors, second_descriptors)
    backward = ratio_matches(second_descriptors, first_descriptors)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    return np.float32([
        [first_keypoints[i].pt for i, _ in pairs],
        [second_keypoints[j].pt for _, j in pairs],
    ]).reshape(2, -1, 2)


def inspect(first, second):
    points = mutual_matches(first[0], second[0])
    shapes = first[1], second[1]
    count = len(points[0])
    upper = [coverage(positions, shape) for positions, shape in zip(points, shapes, strict=True)]
    possible = count >= 12 and min(upper) >= .08
    row = {
        'mutual': count,
        'distinct_locations': [len(np.unique(positions, axis=0)) for positions in points],
        'coverage_upper_bound': upper,
        'could_pass_before_geometry': possible,
        'geometry': None,
        'accepted': False,
    }
    if not possible:
        return row
    cv2.setRNGSeed(0)
    _, mask = cv2.findFundamentalMat(*points, cv2.USAC_MAGSAC, 3., .99, 10_000)
    keep = np.zeros(count, dtype=bool) if mask is None else mask.ravel().astype(bool)
    inliers = int(keep.sum())
    selected = [
        coverage(positions[keep], shape) for positions, shape in zip(points, shapes, strict=True)
    ]
    fraction = inliers / count
    row['geometry'] = {'inliers': inliers, 'fraction': fraction, 'coverage': selected}
    row['accepted'] = inliers >= 12 and fraction >= .50 and min(selected) >= .08
    return row


def run():
    old_path = ROOT / 'docs/research/2026-11-18-synclight-ambient-balance-evidence.json'
    ambient = checked_json(old_path, PIN)['records']
    lamp_dir = BASE / 'synclight-lamp-edit-order-v1'
    lamp = checked_json(lamp_dir / 'results.json', LAMP_REPORT)['records']
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create()
    views = {}
    identity = {}
    for scene, record in lamp.items():
        sources = {
            'target': (
                BASE / 'synclight-ambient-balance' / scene / 'camera2-base.png',
                ambient[scene]['output_sha256']['camera2-base.png'],
            ),
            **{
                arm: (lamp_dir / scene / name, record['hashes'][name])
                for arm, name in [('off', 'std-off-ref.png'), ('on', 'std-on-ref.png')]
            },
        }
        views[scene] = {}
        identity[scene] = {}
        for arm, (path, expected) in sources.items():
            frame = load(path, expected)
            views[scene][arm] = detect(sift, frame), frame.shape
            identity[scene][arm] = expected

    real_pairs, positive, negative = {}, {}, {}
    for scene, frames in views.items():
        other = next(key for key in views if key != scene)
        real_pairs[scene] = {
            arm: inspect(frames['target'], frames[arm]) for arm in ('off', 'on')
        }
        positive[scene] = inspect(frames['off'], frames['on'])
        negative[scene] = inspect(frames['target'], views[other]['on'])
    controls = all(row['accepted'] for row in positive.values()) and not any(
        row['accepted'] for row in negative.values()
    )
    all_real = all(row['accepted'] for room in real_pairs.values() for row in room.values())
    return {
        'opencv': cv2.__version__,
        'input_sha256': identity,
        'cross_view': real_pairs,
        'same_view_controls': positive,
        'different_room_controls': negative,
        'controls_pass': controls,
        'advance_to_colour_fit': controls and all_real,
    }


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = json.dumps(run(), indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(result + '\n')
