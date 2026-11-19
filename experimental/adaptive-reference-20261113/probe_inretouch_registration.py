"""Audit position and profile compatibility of the public example JPEG pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from probe_inretouch_public import STYLE, TARGETS, digest
from reference_color_app.media import decode_image

EVIDENCE = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-inretouch-public-pilot-evidence.json'
RADIUS = 3


class UnmeasurableRegistration(ValueError):
    """At least one required window has no gradient variation."""


def gradient(rgb):
    luma = rgb.astype(np.float64) @ [.299, .587, .114]
    return np.stack([luma[1:, :-1] - luma[:-1, :-1], luma[:-1, 1:] - luma[:-1, :-1]], axis=-1)


def shifts(first, second):
    h, w = first.shape[:2]
    patches = [(RADIUS, h-RADIUS, RADIUS, w-RADIUS)]
    for top, bottom in ((RADIUS, h//2), (h//2, h-RADIUS)):
        for left, right in ((RADIUS, w//2), (w//2, w-RADIUS)):
            patches.append((top, bottom, left, right))
    output = []
    for top, bottom, left, right in patches:
        fixed = first[top:bottom, left:right].ravel()
        x = fixed - fixed.mean()
        scales = np.linalg.norm(x)
        values = []
        for dy in range(-RADIUS, RADIUS+1):
            for dx in range(-RADIUS, RADIUS+1):
                moving = second[top+dy:bottom+dy, left+dx:right+dx].ravel()
                y = moving - moving.mean()
                denominator = scales * np.linalg.norm(y)
                if denominator <= 0:
                    raise UnmeasurableRegistration("No gradient variation in registration window")
                values.append((float(np.dot(x, y) / denominator), dy, dx))
        peak, dy, dx = max(values)
        zero = next(score for score, y, x in values if x == 0 and y == 0)
        output.append(dict(zero=zero, peak=peak, dy=dy, dx=dx, peak_gain=peak-zero))
    return output


def run(source, out):
    out.mkdir(exist_ok=False)
    parent = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    pairs = [(style, f'{style}_Before.jpg', f'{style}_After.jpg') for style in STYLE]
    pairs.extend((target, f'Inference_Data-{target}.jpg', f'GT-{target}.jpg') for target in TARGETS)
    hashes = {entry.removeprefix('example_data/').replace('/', '-'): sha
              for entry, sha in parent['inputs'].items()}
    start = time.monotonic()
    cases = {}
    for label, before, after in pairs:
        metadata = {}
        for filename in (before, after):
            path = source / filename
            assert digest(path) == hashes[filename]
            with Image.open(path) as image:
                profile = image.info.get('icc_profile')
                metadata[filename] = dict(size=list(image.size), mode=image.mode,
                                          orientation=image.getexif().get(274, 1),
                                          icc_sha256=hashlib.sha256(profile).hexdigest() if profile else None)
        original, edited = (decode_image(source / filename) for filename in (before, after))
        assert original.shape == edited.shape
        comparisons = shifts(gradient(original), gradient(edited))
        whole = comparisons[0]
        passed = (whole['zero'] >= .90 and (whole['dy'], whole['dx']) == (0, 0)
                  and all(row['peak_gain'] <= .01 for row in comparisons[1:]))
        profiles_equal = metadata[before]['icc_sha256'] == metadata[after]['icc_sha256']
        cases[label] = dict(metadata=metadata, comparisons=comparisons,
                            no_detected_shift=passed, embedded_profiles_equal=profiles_equal)
        assert time.monotonic() - start < 90
    template = decode_image(source / pairs[0][1])
    control = shifts(gradient(template), gradient(np.roll(template, (-1, 2), axis=(0, 1))))
    assert (control[0]['dy'], control[0]['dx']) == (-1, 2)
    assert time.monotonic() - start < 90
    protocol = Path(__file__).with_name('inretouch-registration-protocol.md')
    report = dict(parent_evidence_sha256=digest(EVIDENCE), script_sha256=digest(__file__),
                  protocol_sha256=digest(protocol), cases=cases, shift_control=control,
                  seconds=time.monotonic()-start)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.source, args.out)
