"""Check whether frozen reference features add illumination-versus-style evidence."""
from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch

from appearance_replication import initialize_replication_model
from probe_beyond_rgb_edit import verified
from probe_beyond_rgb_paired import pinned_report
from probe_beyond_rgb_transfer import digest, write_json
from probe_chromamill_capture_cross import LEVELS, SCENES, decode_rgb8
from probe_physical_blue_global_fit import CAPTURE_SHA, display, linear
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from spatial_render import SpatialEngine


RECEIPT = 'fcc9390ffd82e5627951df539a24bbadf4a16ceb285f1b0dffe0c511f44913db'
BEYOND = '8ba8333f5bd89afd178523459a1dda02d8029d6fb1d4bd297e289c9baaf91d6c'
STATES = ('white', 'physical', 'digital')
TRAIN = ('Scene3_a', 'Scene9_b', 'Scene10_a', 'Scene23_a')


def chromamill_pair(white, blue):
    white_mean, blue_mean = (linear(image).mean(axis=(0, 1)) for image in (white, blue))
    if (white_mean <= 0).any():
        raise ValueError('No chroma sample support')
    shifted = linear(white) * (blue_mean / white_mean)
    edited = display(shifted)
    clip = float((shifted.max(axis=2) > 1).mean())
    error = float(np.max(np.abs(linear(edited).mean(axis=(0, 1)) - blue_mean)))
    return (white, blue, edited), dict(qualified=error <= .02 and clip <= .05,
                                      maximum_mean_rgb_error=error, clip=clip)


def engine_for_probe():
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Unexpected weight')
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    return engine


@torch.inference_mode()
def features(engine, photo):
    reference = engine.analysis(photo)
    pixels = reference[0].cpu().numpy().reshape(3, -1)
    summaries = np.quantile(pixels, [.10, .25, .50, .75, .90], axis=1).ravel()
    color = np.r_[pixels.mean(axis=1), pixels.std(axis=1), summaries]
    image = engine.api['TF'].resize(reference, (64, 64), antialias=True)
    spatial = engine.model.img_projection(image)[0].mean(dim=(-2, -1)).cpu().numpy()
    return dict(color=color, appearance=np.r_[color, spatial])


def read_cases(root):
    assets = root / 'chromamill-correction-training-assets-v1'
    receipt = assets / 'receipt.json'
    if digest(receipt) != RECEIPT:
        raise ValueError('Unexpected training receipt')
    training = json.loads(receipt.read_bytes())['files']
    holdout = root / 'chromamill-capture-cross-v1'
    checked = holdout / 'results.json'
    if digest(checked) != CAPTURE_SHA:
        raise ValueError('Unexpected CHROMA-MILL holdout report')
    approved = json.loads(checked.read_bytes())['prepared']

    for role, scenes in (('train', TRAIN), ('chromamill', SCENES)):
        for scene, level in product(scenes, LEVELS):
            names = [f'{scene}-{hue}-{level}' for hue in ('white', 'blue')]
            frames, checksums = [], []
            for name in names:
                if role == 'train':
                    photo = assets / f'{name}.png'
                    expected = training[name]['sha256']
                    if digest(photo) != expected:
                        raise ValueError('Changed training photo')
                    frames.append(decode_rgb8(photo)[0])
                else:
                    photo = holdout / f'{name}-off.png'
                    expected = approved[scene][name.removeprefix(f'{scene}-')]['hashes']['off']
                    frames.append(verified(photo, expected))
                checksums.append(expected)
            packed, quality = chromamill_pair(*frames)
            yield role, f'{scene}-{level}', scene, packed, quality | {'source_sha256': checksums}

    beyond = root / 'beyond-rgb-matched-cast-first-v1'
    prior = pinned_report(beyond, BEYOND)
    for scene, quality in prior['references'].items():
        names = ('base', 'real', 'edit')
        hashes = [quality['sha256'][name] for name in names]
        packed = tuple(verified(beyond / f'{scene}-guide-{name}.png', sha)
                       for name, sha in zip(names, hashes, strict=True))
        yield 'beyond', scene, scene, packed, dict(qualified=quality['qualified'],
                                                   maximum_mean_rgb_error=None, source_sha256=hashes)


def fit(feature_rows, arm):
    train = [row[arm] for row in feature_rows if row['role'] == 'train' and row['qualified']]
    x = np.array([row[name] for row in train for name in ('physical', 'digital')])
    y = np.tile([1., -1.], len(train))
    center, std = x.mean(axis=0), x.std(axis=0)
    denom = np.maximum(std, 1e-6)
    x = (x - center) / denom / np.sqrt(x.shape[1])
    dual = np.linalg.solve(x @ x.T + .1 * np.eye(len(x)), y)
    weights = x.T @ dual
    return center, denom, weights


def score(feature_rows, arm):
    center, denom, weights = fit(feature_rows, arm)
    rows = {}
    for row in feature_rows:
        scores = {name: float(((x - center) / denom) @ weights / np.sqrt(len(weights)))
                  for name, x in row[arm].items()}
        rows[row['key']] = dict(role=row['role'], qualified=row['qualified'], scores=scores,
                               pair_correct=scores['physical'] > 0 and scores['digital'] <= 0,
                               white_false_trigger=scores['white'] > 0)
    return rows


def run(root, out):
    model = engine_for_probe()
    archive, rows = {}, []
    for role, key, scene, frames, metadata in read_cases(root):
        archive[key] = dict(role=role, scene=scene, **metadata)
        row = dict(role=role, key=key, qualified=metadata['qualified'])
        extracts = [features(model, photo) for photo in frames]
        row.update({arm: dict(zip(STATES, [v[arm] for v in extracts], strict=True)) for arm in extracts[0]})
        rows.append(row)

    fit_sources = [row for row in archive.values() if row['role'] == 'train' and row['qualified']]
    fit_ok = len(fit_sources) >= 6 and len({row['scene'] for row in fit_sources}) >= 3
    predictions = {arm: score(rows, arm) for arm in ('color', 'appearance')} if fit_ok else {}
    out.mkdir()
    write_json(out / 'report.json', dict(appearance_sha256=WEIGHT_SHA256, inputs=archive,
                                        fit_qualified=fit_ok, predictions=predictions))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--opencv-site', type=Path)
    args = parser.parse_args()
    if args.opencv_site is not None:
        sys.path.append(str(args.opencv_site.resolve()))
    run(args.root, args.out)
