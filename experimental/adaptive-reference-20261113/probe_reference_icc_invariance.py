"""Check whether equivalent sRGB and Display P3 references yield the same grade."""

import argparse
from io import BytesIO
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageCms
import torch

from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import (
    BASE, BG_SHA, CHECKPOINT, LUMA, STRENGTH, SpatialEngine, checked_json, checked_rgb, chroma,
    digest, json_to, panel, save,
)
from reference_color_app.media import decode_image
from spatial_render import finish_spatial

PROFILE_SHA = 'cb51de38e482ee974c0c76b9689e16aad04bad16e226fed2f30c842d15ff3a3d'
INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC


def convert(image, source, dest):
    return ImageCms.profileToProfile(image, source, dest, renderingIntent=INTENT, outputMode='RGB')


def read_normalized(path, target_profile):
    with Image.open(path) as frame:
        profile = frame.info.get('icc_profile')
        return (np.array(convert(frame, ImageCms.ImageCmsProfile(BytesIO(profile)), target_profile))
                if profile else decode_image(path))


def srgb_profile():
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    # ICC header creation date is metadata. Fix it so the tagged control is reproducible.
    struct.pack_into('>6H', profile, 24, 2026, 11, 20, 0, 0, 0)
    return ImageCms.ImageCmsProfile(BytesIO(profile))


def render(engine, target, guide):
    return finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)


def measure(source, folder, inputs, boxes, engine, srgb, p3):
    folder.mkdir()
    target = checked_rgb(source / 'target.png', inputs['target.png'])
    expected = checked_rgb(source / 'expected-on.png', inputs['expected-on.png'])
    rows, board, renders = {}, {}, {}
    for group in ('C1', 'F1'):
        renders[group] = {}
        for state in ('off', 'on'):
            tag = f'{group}-{state}'
            original = checked_rgb(source / f'{tag}-reference.png', inputs[f'{tag}-reference.png'])
            baseline = checked_rgb(source / f'{tag}-13.png', inputs[f'{tag}-13.png'])
            np.testing.assert_array_equal(render(engine, target, original), baseline)
            with Image.open(source / f'{tag}-reference.png') as frame:
                if frame.info.get('icc_profile'):
                    raise ValueError('Baseline must be the frozen untagged sRGB working image')
            tagged = folder / f'{tag}-p3.png'
            reencoded = convert(Image.fromarray(original), srgb, p3)
            reencoded.save(tagged, icc_profile=p3.tobytes())
            old = render(engine, target, decode_image(tagged))
            new_reference = read_normalized(tagged, srgb)
            new = render(engine, target, new_reference)
            renders[group][state] = {'old': old, 'new': new, 'baseline': baseline}
            variants = {}
            for label, frame in (('old', old), ('new', new)):
                delta = frame.astype(float) - baseline
                color = chroma(delta, delta @ LUMA)
                variants[label] = {'whole_mae': float(np.abs(delta).mean()),
                                   'region_chroma_mae': {name: float(np.abs(color[area]).mean()) for name, area in boxes.items()},
                                   'sha256': save(folder / f'{tag}-{label}.png', frame)}
            rows[tag] = dict(frames=variants, tagged_sha256=digest(tagged),
                             decoded_roundtrip_mae=float(np.abs(new_reference.astype(float) - original).mean()))
            board[tag] = {'original sRGB input': original, '13 / original': baseline, 'P3 / ignored': old,
                          'P3 / managed': new, 'known edit': expected if state == 'on' else target}
    edit = expected.astype(float) - target
    norm = np.sum(edit ** 2)
    if not norm:
        raise ValueError('Known edit has no signal')
    projections = {group: {arm: float(np.sum((ends['on'][arm].astype(float) - ends['off'][arm]) * edit) / norm)
                           for arm in ('baseline', 'old', 'new')} for group, ends in renders.items()}
    return dict(rows=rows, projections=projections, comparison_sha256=panel(folder / 'comparison.png', board))


@torch.inference_mode()
def run(out):
    source = BASE / 'bg-real-editor-look'
    manifest = checked_json(source / 'results.json', BG_SHA)
    path = BASE / 'icc-reference-assets-v1' / 'DisplayP3-v4.icc'
    if path.stat().st_size != 480 or digest(path) != PROFILE_SHA:
        raise ValueError('Unexpected Display P3 profile')
    p3 = ImageCms.ImageCmsProfile(BytesIO(path.read_bytes()))
    srgb = srgb_profile()
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Unexpected 13 profile')
    out.mkdir()
    sample = np.random.default_rng(215).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    for tag, opts in (('untagged', {}), ('srgb', {'icc_profile': srgb.tobytes()})):
        file = out / f'{tag}.png'
        Image.fromarray(sample).save(file, **opts)
        np.testing.assert_array_equal(decode_image(file), sample)
        np.testing.assert_array_equal(read_normalized(file, srgb), sample)
    boxes = regions()
    cases = {pair: measure(source / pair, out / pair, row['artifacts'], boxes[pair], engine, srgb, p3)
             for pair, row in manifest['cases'].items()}
    observations = [row for scene in cases.values() for row in scene['rows'].values()]
    drift = [row for row in observations if row['frames']['old']['whole_mae'] >= .5]
    restored = sum(row['frames']['new']['whole_mae'] <= min(1, .5 * row['frames']['old']['whole_mae']) for row in drift)
    worst_roundtrip = max(row['decoded_roundtrip_mae'] for row in observations)
    if len(observations) != 16:
        raise ValueError('Unexpected image count')
    json_to(out / 'results.json', dict(checkpoint_sha256=CHECKPOINT, parent_sha256=BG_SHA, profile_sha256=PROFILE_SHA,
                                      cases=cases, wrong_decode_drift=len(drift), restored=restored,
                                      worst_reference_roundtrip=worst_roundtrip,
                                      passed=len(drift) >= 12 and restored >= 12 and worst_roundtrip <= 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
