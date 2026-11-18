"""Compare 13's color response to a labeled, fixed-camera red/blue light swap."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch

from audit_fau_two_light_intake import ARCHIVE_SHA, LAB, native_rgb, sha
from background_intervention import target_regions
from independent_coco import digest
from probe_multilum_chromaticity import differences
from spatial_render import SpatialEngine, finish_spatial


SCENES = ('bmug', 'bpmug')


def member(archive, role, prefix, lighting):
    name = f'{LAB}/{role}/{prefix}_{lighting}.png'
    entry = archive.getmember(name)
    assert entry.isfile() and 0 < entry.size <= 2*1024**2
    data = archive.extractfile(entry).read()
    assert len(data) == entry.size
    return data


def encode(raw):
    linear = raw.astype(np.float64) / 65535
    srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return np.rint(srgb * 255).astype(np.uint8)


def chroma_change(start, end):
    delta = np.log(end.astype(float)) - np.log(start.astype(float))
    return delta[:, [0, 2]] - delta[:, 1:2]


def evaluate(region, valid, raw, labels, target, reference, result):
    selected = region & valid
    fraction = float(selected.sum()/region.sum()) if region.any() else 0.
    if not selected.any():
        return dict(eligible=False, fraction=fraction, pixels=0)
    actual = chroma_change(raw[0][selected], raw[1][selected])
    light = chroma_change(labels[0][selected], labels[1][selected])
    denominator = float(np.linalg.norm(actual)*np.linalg.norm(light))
    agreement = float(np.sum(actual * light) / denominator) if denominator else 0.
    desired, delta_y = differences(target[selected], reference[selected])
    observed, output_y = differences(target[selected], result[selected])
    power = float(np.sum(desired**2))
    assert power > 0
    rms = float(np.sqrt(np.mean(desired**2)))
    rms_error = float(np.sqrt(np.mean((desired-observed)**2)))
    projection = float(np.sum(desired*observed) / power)
    eligible = bool(fraction >= .20 and rms >= .05 and agreement >= .60)
    return dict(eligible=eligible, fraction=fraction, pixels=int(selected.sum()), label_agreement=agreement,
                actual_chroma_rms=rms, output_chroma_rms_error=rms_error, output_projection=projection,
                actual_lightness_rms=float(np.sqrt(np.mean(delta_y**2))),
                output_lightness_rms=float(np.sqrt(np.mean(output_y**2))),
                passed=bool(eligible and projection >= .50 and rms_error <= .90*rms))


def run(path, out):
    assert digest(path) == ARCHIVE_SHA
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    out.mkdir()
    cases, saved = {}, {}
    with tarfile.open(path, 'r:gz') as archive:
        pairs = {}
        for scene in SCENES:
            data = [member(archive, 'img', scene, side) for side in ('b_r', 'r_b')]
            raw = [native_rgb(content, 260, 452) for content in data]
            target, reference = map(encode, raw)
            output = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
            pairs[scene] = (raw, target, reference, output)
            saved[scene] = {}
            for role, frame in dict(target=target, reference=reference, current13=output).items():
                filename = out / f'{scene}-{role}.png'
                Image.fromarray(frame).save(filename)
                saved[scene][role] = digest(filename)
            cases[scene] = dict(raw_sha256=[sha(content) for content in data])
        for scene, (raw, target, reference, output) in pairs.items():
            arrays = {}
            for role in ('groundtruth', 'masks'):
                data = [member(archive, role, scene, side) for side in ('b_r', 'r_b')]
                arrays[role] = [np.array(Image.open(io.BytesIO(content))) for content in data]
                cases[scene][f'{role}_sha256'] = [sha(content) for content in data]
            labels = arrays['groundtruth']
            valid = (arrays['masks'][0] == 255) & (arrays['masks'][1] == 255)
            for image, label in zip((target, reference), labels):
                valid &= ((image >= 8) & (image <= 247) & (label > 0)).all(axis=2)
            regions = target_regions(target, np.zeros(valid.shape, bool))
            desired = finish_spatial(target, reference, .8)
            cases[scene]['scores'] = {name: evaluate(regions[name], valid, raw, labels, target, desired, output)
                                      for name in ('whole', 'neutral_proxy')}
    report = dict(cases=cases, predictions=saved, archive_sha256=ARCHIVE_SHA, checkpoint_sha256=engine.checkpoint_sha,
                  source_sha256=digest(Path(__file__)))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({scene: row['scores'] for scene,row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
