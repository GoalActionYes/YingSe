"""Screen fixed lab-light setting recognition on other FAU objects."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from audit_fau_two_light_intake import ARCHIVE_SHA, LAB, native_rgb
from independent_coco import ROOT, digest
from probe_fau_estimated_light import predictor
from probe_fau_two_light_response import SCENES, encode, member
from probe_multilum_chromaticity import linear_rgb
from reference_color_app.media import decode_image


SETTINGS = ('b_r', 'r_b')
MODES = ('rgb', 'illumination')


def left_right(image):
    log_rgb = np.log(np.maximum(image.astype(float), 1e-4))
    chroma = log_rgb[..., [0, 2]] - log_rgb[..., 1:2]
    half = chroma.shape[1] // 2
    return chroma[:, :half].mean(axis=(0, 1)) - chroma[:, half:].mean(axis=(0, 1))


def run(root, out):
    archive_path = root / 'benchmark_mirf.tar.gz'
    assert digest(archive_path) == ARCHIVE_SHA
    docs = ROOT / 'docs/research'
    exposure = docs / '2026-11-18-fau-exposure-evidence.json'
    illumination = docs / '2026-11-18-fau-estimated-light-evidence.json'
    inputs, earlier = (json.loads(path.read_bytes()) for path in (exposure, illumination))
    assert earlier['exposure_sha256'] == digest(exposure)

    calibration = {}
    for scene, (setting, role) in product(SCENES, zip(SETTINGS, ('target', 'reference'))):
        rgb = root / 'exposure' / f'{scene}-{role}.png'
        shade = root / 'estimated-light' / f'{scene}-{role}-shade.npy'
        assert digest(rgb) == inputs['predictions'][scene][role]
        assert digest(shade) == earlier['cases'][scene]['shading'][role]
        calibration[scene, setting] = dict(
            rgb=left_right(linear_rgb(decode_image(rgb))),
            illumination=left_right(np.load(shade, allow_pickle=False)),
        )
    centers = {mode: np.array([np.mean([calibration[scene, setting][mode] for scene in SCENES], axis=0)
                                for setting in SETTINGS]) for mode in MODES}
    predict = predictor()
    out.mkdir()
    with tarfile.open(archive_path, 'r:gz') as archive:
        frames = {Path(item.name).stem for item in archive if item.isfile() and
                  item.name.startswith(f'{LAB}/img/')}
        buckets = [{key.removesuffix('_'+setting) for key in frames if key.endswith('_'+setting)}
                   for setting in SETTINGS]
        scenes = sorted(scene for scene in set.intersection(*buckets) if 'mug' not in scene)
        assert len(scenes) == 5 and not set(scenes).intersection(SCENES)
        cases = {}
        for index, (scene, setting) in enumerate(product(scenes, SETTINGS)):
            raw = member(archive, 'img', scene, setting)
            photo = encode((native_rgb(raw, 260, 452).astype(np.uint32)*4).clip(0, 65535))
            input_path = out / f'{index:02d}-rgb.png'
            shade_path = out / f'{index:02d}-shade.npy'
            Image.fromarray(photo).save(input_path)
            shade = predict(photo)[1]
            np.save(shade_path, shade, allow_pickle=False)
            variants = {}
            for mode, feature in (('rgb', left_right(linear_rgb(photo))), ('illumination', left_right(shade))):
                distances = np.linalg.norm(centers[mode] - feature, axis=1)
                variants[mode] = dict(feature=feature.tolist(), distances=distances.tolist(),
                                      predicted=SETTINGS[int(np.argmin(distances))])
            cases[scene, setting] = dict(index=index, variants=variants, rgb_sha256=digest(input_path),
                                         shade_sha256=digest(shade_path))

    summary = {}
    for mode in MODES:
        correct = sum(row['variants'][mode]['predicted'] == setting for (_, setting), row in cases.items())
        errors = dict(same=0, changed=0)
        for first, second in combinations(scenes, 2):
            for left, right in product(SETTINGS, repeat=2):
                left_guess = cases[first, left]['variants'][mode]['predicted']
                right_guess = cases[second, right]['variants'][mode]['predicted']
                if (left_guess == right_guess) != (left == right):
                    errors['same' if left == right else 'changed'] += 1
        summary[mode] = dict(correct=correct, pair_errors=errors,
                             passed=correct == len(cases) and not any(errors.values()))

    report = dict(archive_sha256=ARCHIVE_SHA, source_sha256=digest(Path(__file__)),
                  exposure_sha256=digest(exposure), illumination_sha256=digest(illumination),
                  centers={mode: value.tolist() for mode, value in centers.items()},
                  cases={f'{scene}_{setting}': row for (scene, setting), row in cases.items()},
                  summary=summary, proceed=summary['illumination']['passed'] and not summary['rgb']['passed'])
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.root, args.out)
