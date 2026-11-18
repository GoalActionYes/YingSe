"""Separate fixed RGB reconstruction error from cross-scene transfer error."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_s24_donor_delta import decode_image, digest, finish_spatial, image, matrix, render, target_regions


REGIONS = ('whole', 'neutral_proxy')


def held_out_regions(source):
    pixels = np.ones(source.shape[:2], dtype=bool)
    pixels[::8, ::8] = False
    regions = target_regions(source, np.zeros_like(pixels))
    return {key: pixels & regions[key] for key in REGIONS}


def errors(frame, desired, regions):
    delta = np.abs(frame.astype(np.int16) - desired)
    return {key: float(delta[mask].mean()) for key, mask in regions.items()}


def classify(cross, donor, target):
    excess = cross - max(donor, target)
    if excess >= 5:
        return 'transfer_excess'
    if excess <= 2:
        return 'comparable_within_scene_error'
    return 'indeterminate'


def run(assets, evidence_path, previous, out):
    prior = json.loads(evidence_path.read_bytes())
    assert digest(assets / 'assets.json') == prior['assets_sha256']
    assert digest(previous / 'results.json') == prior['reproduction']['identical_report_sha256']
    fitter = Path(__file__).with_name('probe_s24_donor_delta.py')
    assert digest(fitter) == prior['source_sha256']
    manifest = json.loads((assets / 'assets.json').read_bytes())
    out.mkdir()

    def load(scene, grade):
        return decode_image(image(assets, manifest, scene, f'srgb_images_style_{grade}'))

    within, hashes = {}, {}
    for scene in prior['cases']:
        source = load(scene, 1)
        regions = held_out_regions(source)
        within[scene] = {}
        for grade in (2, 3):
            authored = load(scene, grade)
            fitted = render(source, matrix(source, authored))
            path = out / f'{scene}-self-{grade}.png'
            Image.fromarray(fitted).save(path)
            hashes[path.name] = digest(path)
            within[scene][grade] = errors(fitted, finish_spatial(source, authored, .8), regions)

    slots = []
    for scene, record in prior['cases'].items():
        source = load(scene, 1)
        regions = held_out_regions(source)
        for slot, (donor, grade) in enumerate(record['references']):
            path = previous / f'{scene}-paired-{slot}.png'
            assert digest(path) == prior['predictions'][scene][path.name]
            cross = errors(decode_image(path), finish_spatial(source, load(scene, grade), .8), regions)
            donor_self, target_self = within[donor][grade], within[scene][grade]
            readings = {}
            for key in REGIONS:
                readings[key] = dict(cross=cross[key], donor_self=donor_self[key], target_self=target_self[key],
                                     result=classify(cross[key], donor_self[key], target_self[key]))
            slots.append(dict(scene=scene, donor=donor, grade=grade, slot=slot, readings=readings))

    report = dict(excluded=prior['excluded'], previous_evidence_sha256=digest(evidence_path),
                  source_sha256=digest(Path(__file__)), self_sha256=hashes, slots=slots)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: sum(s['readings'][key]['result'] == 'transfer_excess' for s in slots)
                      for key in REGIONS}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('assets', 'evidence_path', 'previous', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
