"""Compare the existing spatial renderer with paired real light changes."""
import argparse
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial

SCENES = {
    'main_experiment120': 'fa4f368ac11170f6ebacf3aaac4af074961b7b9f03e09c2fb060056fa8862b48',
    'kingston_dining10': '302e5089ac541e5da5f585507ff55fb8319fe242cc538e148711429936a8d104',
}
DIRECTIONS = (14, 24)
STRENGTH = .8


def run(archives, out):
    out.mkdir(exist_ok=False)
    images = {}
    inputs = {}
    for scene, expected in SCENES.items():
        archive_path = archives / f'{scene}.zip'
        assert digest(archive_path) == expected
        images[scene], inputs[scene] = {}, {}
        with ZipFile(archive_path) as archive:
            for direction in DIRECTIONS:
                member = f'{scene}/dir_{direction}_mip2.jpg'
                assert archive.namelist().count(member) == 1
                assert archive.getinfo(member).file_size < 2 * 1024 * 1024
                path = out / f'{scene}-{direction}.jpg'
                path.write_bytes(archive.read(member))
                image = decode_image(path)
                assert image.shape == (1000, 1500, 3)
                inputs[scene][direction] = digest(path)
                images[scene][direction] = image

    engine = SpatialEngine('multiref-v1')
    start = time.monotonic()
    paired = [(target, next(scene for scene in SCENES if scene != target)) for target in SCENES]
    outputs = {}
    for target, reference in paired:
        original = images[target][14]
        outputs[target] = {}
        for direction in DIRECTIONS:
            grade = engine.prepare(original, images[reference][direction])
            result = finish_spatial(original, grade.apply(original), STRENGTH)
            path = out / f'{target}-output-ref{direction}.png'
            Image.fromarray(result).save(path)
            outputs[target][direction] = digest(path)
            assert time.monotonic() - start < 90

    protocol = Path(__file__).with_name('multilum-pilot-protocol.md')
    report = dict(model_sha256=engine.checkpoint_sha, strength=STRENGTH, archives=SCENES,
                  inputs=inputs, outputs=outputs, render_and_save_seconds=time.monotonic()-start,
                  protocol_sha256=digest(protocol), script_sha256=digest(__file__))
    (out / 'predictions.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    cases = []
    for target, reference in paired:
        actual = images[target][24].astype(np.int16) - images[target][14]
        with Image.open(out / f'{target}-output-ref14.png') as first:
            baseline = np.asarray(first, dtype=np.int16)
        with Image.open(out / f'{target}-output-ref24.png') as second:
            change = np.asarray(second, dtype=np.int16) - baseline
        neutral = target_regions(images[target][14], np.zeros(actual.shape[:2], bool))['neutral_proxy']
        result = {}
        for region, mask in [('whole', np.ones_like(neutral)), ('neutral_proxy', neutral)]:
            a, b = actual[mask].astype(np.float64), change[mask].astype(np.float64)
            observed_mae = float(np.abs(a).mean()) if a.size else None
            usable = bool(mask.mean() >= .01 and observed_mae is not None and observed_mae >= .5)
            if usable:
                error = float(np.abs(a-b).mean())
                projection = float(np.vdot(a, b) / np.vdot(a, a))
            else:
                error = projection = None
            result[region] = dict(pixel_fraction=float(mask.mean()), usable=usable,
                                  observed_mae=observed_mae, generated_mae=float(np.abs(b).mean()) if b.size else None,
                                  response_error=error, response_projection=projection,
                                  better_than_no_response=error < observed_mae if usable else None)
        cases.append(dict(target=target, reference=reference, metrics=result))
    report.update(cases=cases, predictions_sha256=digest(out / 'predictions.json'))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.archives, args.out)
