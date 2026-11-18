"""Recheck red/blue light response with an identical two-stop exposure increase."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch

from audit_fau_two_light_intake import ARCHIVE_SHA, native_rgb, sha
from background_intervention import target_regions
from independent_coco import digest
from probe_fau_two_light_response import SCENES, encode, evaluate, member
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


def bounds_mask(images):
    return np.logical_and.reduce([((image >= 8) & (image <= 247)).all(axis=-1) for image in images])


def run(path, prior_evidence, previous, out):
    assert digest(path) == ARCHIVE_SHA
    prior = json.loads(prior_evidence.read_bytes())['response']
    assert prior['archive_sha256'] == ARCHIVE_SHA
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == prior['checkpoint_sha256']
    out.mkdir()
    cases, renders, inputs = {}, {}, {}
    with tarfile.open(path, 'r:gz') as archive:
        for scene in SCENES:
            data = [member(archive, 'img', scene, side) for side in ('b_r', 'r_b')]
            assert [sha(item) for item in data] == prior['cases'][scene]['raw_sha256']
            raw = [native_rgb(item, 260, 452) for item in data]
            scaled = [array.astype(np.uint32)*4 for array in raw]
            bright = [encode(value.clip(0, 65535)) for value in scaled]
            target, reference = bright
            output = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
            renders[scene] = {}
            for role, image in zip(('target', 'reference', 'current13'), (target, reference, output)):
                file = out / f'{scene}-{role}.png'
                Image.fromarray(image).save(file)
                renders[scene][role] = digest(file)
            old = []
            for role in ('target', 'reference', 'current13'):
                file = previous / f'{scene}-{role}.png'
                assert digest(file) == prior['predictions'][scene][role]
                old.append(decode_image(file))
            assert all(np.array_equal(encoded, encode(array)) for encoded, array in zip(old[:2], raw))
            inputs[scene] = (raw, old, (target, reference, output))
            cases[scene] = dict(clipped_pixels=[int((value > 65535).any(axis=-1).sum()) for value in scaled])

        for scene, (raw, old, bright) in inputs.items():
            annotations = {}
            for role in ('masks', 'groundtruth'):
                data = [member(archive, role, scene, direction) for direction in ('b_r', 'r_b')]
                assert [sha(item) for item in data] == prior['cases'][scene][f'{role}_sha256']
                annotations[role] = [np.array(Image.open(io.BytesIO(item))) for item in data]
            labels = annotations['groundtruth']
            allowed = np.logical_and.reduce([mask == 255 for mask in annotations['masks']]
                                            + [(label > 0).all(axis=-1) for label in labels])
            bright_valid = allowed & bounds_mask(bright[:2])
            old_valid = allowed & bounds_mask(old[:2])
            common = bright_valid & old_valid

            def grades(frames, valid, region_source):
                target, reference, result = frames
                regions = target_regions(region_source, np.zeros(valid.shape, bool))
                desired = finish_spatial(target, reference, .8)
                return {r: evaluate(regions[r], valid, raw, labels, target, desired, result)
                        for r in ('whole', 'neutral_proxy')}

            cases[scene].update(bright=grades(bright, bright_valid, bright[0]),
                                common_old=grades(old, common, old[0]), common_bright=grades(bright, common, old[0]))
    report = dict(cases=cases, predictions=renders, archive_sha256=ARCHIVE_SHA, output_gain=4,
                  prior_sha256=digest(prior_evidence), checkpoint_sha256=engine.checkpoint_sha,
                  source_sha256=digest(Path(__file__)))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for scene, row in cases.items():
        print(scene, {mode: {region: (entry.get('eligible'), entry.get('output_projection'),
                                      entry.get('output_chroma_rms_error'), entry.get('fraction'), entry.get('passed'))
                             for region, entry in row[mode].items()} for mode in ('bright','common_old','common_bright')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('path', 'prior_evidence', 'previous', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
