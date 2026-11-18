"""Calibrate a target-only light response on one scene and render the other."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from audit_fau_two_light_intake import ARCHIVE_SHA, native_rgb
from background_intervention import target_regions
from independent_coco import ROOT, digest
from probe_fau_exposure_control import bounds_mask
from probe_fau_illuminant_replay import replay
from probe_fau_two_light_response import SCENES, chroma_change, evaluate, member
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def design(shade):
    channels = np.log(np.maximum(shade.astype(float), 1e-4))
    chroma = channels[..., [0, 2]] - channels[..., 1:2]
    return np.concatenate((np.ones((*shade.shape[:2], 1)), chroma), axis=-1)


def labeled_area(archive, scene, frames):
    labels = [np.array(Image.open(io.BytesIO(member(archive, 'groundtruth', scene, side))))
              for side in ('b_r', 'r_b')]
    valid = bounds_mask(frames)
    for label, side in zip(labels, ('b_r', 'r_b')):
        mask = np.array(Image.open(io.BytesIO(member(archive, 'masks', scene, side))))
        valid &= (mask == 255) & (label > 0).all(axis=-1)
    return labels, valid


def render(target, delta):
    shape = delta.shape[:-1]
    rgb_ratio = np.exp(np.stack((delta[..., 0], np.zeros(shape), delta[..., 1]), axis=-1))
    assert np.isfinite(rgb_ratio).all()
    return replay(target, rgb_ratio.reshape(-1, 3), np.ones(target.shape[:2], bool))[0]


def run(root, out):
    archive_path = root / 'benchmark_mirf.tar.gz'
    assert digest(archive_path) == ARCHIVE_SHA
    docs = ROOT / 'docs/research'
    exposure = docs / '2026-11-18-fau-exposure-evidence.json'
    light = docs / '2026-11-18-fau-estimated-light-evidence.json'
    inputs, model = (json.loads(path.read_bytes()) for path in (exposure, light))
    assert model['exposure_sha256'] == digest(exposure)
    frames, features = {}, {}
    for scene in SCENES:
        frames[scene] = []
        for role in ('target', 'reference'):
            photo = root / 'exposure' / f'{scene}-{role}.png'
            assert digest(photo) == inputs['predictions'][scene][role]
            frames[scene].append(decode_image(photo))
        array = root / 'estimated-light' / f'{scene}-target-shade.npy'
        assert digest(array) == model['cases'][scene]['shading']['target']
        features[scene] = design(np.load(array, allow_pickle=False))
        target = frames[scene][0]
        assert np.array_equal(render(target, np.zeros((*target.shape[:2], 2))), target)

    out.mkdir()
    cases, generated = {}, {}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for train, test in (SCENES, SCENES[::-1]):
            labels, valid = labeled_area(archive, train, frames[train])
            delta = chroma_change(labels[0][valid], labels[1][valid])
            fit, _, rank, _ = np.linalg.lstsq(features[train][valid], delta, rcond=None)
            assert rank == 3
            target = frames[test][0]
            mean = np.broadcast_to(delta.mean(axis=0), (*target.shape[:2], 2))
            outputs = {'constant': render(target, mean), 'target_light': render(target, features[test] @ fit)}
            records = {}
            for name, photo in outputs.items():
                path = out / f'{test}-{name}.png'
                Image.fromarray(photo).save(path)
                records[name] = {'sha256': digest(path)}
            generated[test] = outputs
            cases[test] = dict(calibration_scene=train, calibration_pixels=int(valid.sum()),
                               coefficients=fit.tolist(), mean_delta=delta.mean(axis=0).tolist(),
                               outputs=records)

        for test, row in cases.items():
            target, reference = frames[test]
            labels, valid = labeled_area(archive, test, frames[test])
            raw = [native_rgb(member(archive, 'img', test, side), 260, 452) for side in ('b_r', 'r_b')]
            desired = finish_spatial(target, reference, .8)
            regions = target_regions(target, np.zeros(valid.shape, bool))
            for name, record in row['outputs'].items():
                record['regions'] = {
                    area: evaluate(regions[area], valid, raw, labels, target, desired, generated[test][name])
                    for area in ('whole', 'neutral_proxy')
                }
            candidate, constant = (row['outputs'][name]['regions'] for name in ('target_light', 'constant'))
            metric = 'output_chroma_rms_error'
            row['neutral_gain'] = 1 - candidate['neutral_proxy'][metric] / constant['neutral_proxy'][metric]
            row['passed'] = row['neutral_gain'] >= .15 and all(region['passed'] for region in candidate.values())

    report = dict(cases=cases, proceeded=all(row['passed'] for row in cases.values()),
                  exposure_evidence_sha256=digest(exposure), lighting_evidence_sha256=digest(light),
                  source_sha256=digest(Path(__file__)), archive_sha256=ARCHIVE_SHA, identity_passed=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.root, args.out)
