"""Replay measured local versus global light changes on the frozen FAU pair."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from audit_fau_two_light_intake import ARCHIVE_SHA, native_rgb, sha
from background_intervention import target_regions
from independent_coco import digest
from probe_fau_exposure_control import bounds_mask
from probe_fau_two_light_response import SCENES, encode, evaluate, member
from probe_multilum_chromaticity import linear_rgb
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


LUMA = np.array([.2126, .7152, .0722])


def replay(target, gain, mask):
    original = linear_rgb(target)
    changed = original[mask]*gain
    brightness = original[mask] @ LUMA
    denominator = changed @ LUMA
    adjusted = changed * np.divide(brightness, denominator, out=np.ones_like(brightness),
                                    where=denominator > 0)[:, None]
    clip = int((adjusted > 1).any(axis=1).sum())
    full = target.copy()
    full[mask] = encode(adjusted.clip(0, 1)*65535)
    return finish_spatial(target, full, .8), clip


def run(path, exposure, original, inputs, out):
    assert digest(path) == ARCHIVE_SHA
    exposure_report = json.loads(exposure.read_bytes())
    original_report = json.loads(original.read_bytes())['response']
    assert exposure_report['prior_sha256'] == digest(original)
    assert exposure_report['archive_sha256'] == original_report['archive_sha256'] == ARCHIVE_SHA
    out.mkdir()
    cases = {}
    with tarfile.open(path, 'r:gz') as archive:
        for scene in SCENES:
            frames = {}
            for role, expected in exposure_report['predictions'][scene].items():
                file = inputs / f'{scene}-{role}.png'
                assert digest(file) == expected
                frames[role] = decode_image(file)
            target, reference = frames['target'], frames['reference']
            arrays = {}
            for role in ('img', 'masks', 'groundtruth'):
                data = [member(archive, role, scene, side) for side in ('b_r', 'r_b')]
                digest_key = 'raw_sha256' if role == 'img' else f'{role}_sha256'
                assert [sha(item) for item in data] == original_report['cases'][scene][digest_key]
                if role == 'img':
                    arrays[role] = [native_rgb(item, 260, 452) for item in data]
                else:
                    arrays[role] = [np.array(Image.open(io.BytesIO(item))) for item in data]
            labels = arrays['groundtruth']
            mask = bounds_mask((target, reference))
            for included, label in zip(arrays['masks'], labels):
                mask &= (included == 255) & (label > 0).all(axis=-1)
            assert mask.any()
            ratio = labels[1][mask].astype(float) / labels[0][mask]
            ratio /= ratio[:, 1:2]
            mean = np.exp(np.log(ratio).mean(axis=0))
            assert np.array_equal(replay(target, np.ones(3), mask)[0], target)
            desired = finish_spatial(target, reference, .8)
            regions = target_regions(target, np.zeros(mask.shape, bool))
            record = {}
            for mode, gain in (('global', mean), ('local', ratio)):
                grade, clipped = replay(target, gain, mask)
                assert np.array_equal(grade[~mask], target[~mask])
                file = out / f'{scene}-{mode}.png'
                Image.fromarray(grade).save(file)
                assert grade.shape == target.shape
                scores = {key: evaluate(regions[key], mask, arrays['img'], labels, target, desired, grade)
                          for key in ('whole', 'neutral_proxy')}
                luma_delta = (linear_rgb(grade)[mask]-linear_rgb(target)[mask]) @ LUMA
                record[mode] = dict(file_sha256=digest(file), clipped=clipped, scores=scores,
                                    linear_luma_mae=float(np.abs(luma_delta).mean()))
            a, b = [record[mode]['scores']['neutral_proxy']['output_chroma_rms_error'] for mode in ('global', 'local')]
            record['neutral_gain'] = float(1-b/a)
            record['local_pass'] = record['neutral_gain'] >= .15 and all(
                entry['passed'] for entry in record['local']['scores'].values())
            cases[scene] = record
    report = dict(cases=cases, proceeded=all(row['local_pass'] for row in cases.values()),
                  exposure_sha256=digest(exposure), original_sha256=digest(original),
                  archive_sha256=ARCHIVE_SHA, source_sha256=digest(Path(__file__)))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('path', 'exposure', 'original', 'inputs', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
