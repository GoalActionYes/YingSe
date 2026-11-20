"""Audit Canon capture gains and their persistence through a real pixel edit."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from struct import calcsize, unpack_from

import numpy as np
from PIL import Image

from probe_gros_portrait_light_transfer import GROUPS, PHOTO_SHA
from probe_gros_screen_chroma import ARCHIVE, face_weight, local_edit
from probe_target_mirror_chroma import checked_json, digest, json_to
from triplet_color import to_srgb


def capture_gains(image):
    directory = image.getexif()
    exif = image.info.get('exif', b'')
    maker = directory.get_ifd(34665).get(37500, b'')
    if directory.get(272) != 'Canon EOS REBEL T2i' or not exif.startswith(b'Exif\x00\x00') or len(maker) < 2:
        return None
    tiff = exif[6:]
    order = {b'II': '<', b'MM': '>'}.get(tiff[:2])
    if order is None or len(tiff) < 8 or unpack_from(order + 'H', tiff, 2)[0] != 42:
        return None
    size = calcsize(order + 'HHII')
    entries = unpack_from(order + 'H', maker)[0]
    if 2 + entries * size > len(maker):
        return None
    expected = []
    for index in range(entries):
        tag, kind, count, offset = unpack_from(order + 'HHII', maker, 2 + size * index)
        if tag == 0x4001:
            expected.append((kind, count, offset))
    if len(expected) != 1:
        return None
    kind, count, offset = expected[0]
    if (kind, count) != (3, 1338) or not 8 <= offset <= len(tiff) - count * 2:
        return None
    if unpack_from(order + 'h', tiff, offset)[0] != 7:
        return None
    shot = list(unpack_from(order + '4h', tiff, offset + 2 * 0x3f))
    auto = list(unpack_from(order + '4h', tiff, offset + 2 * 0x44))
    if min(shot + auto) <= 0:
        return None
    return dict(shot_rggb=shot, auto_rggb=auto,
                shot_temperature=unpack_from(order + 'h', tiff, offset + 2 * 0x43)[0],
                auto_temperature=unpack_from(order + 'h', tiff, offset + 2 * 0x48)[0])


def normalized(values):
    red, green_a, green_b, blue = values
    return np.log(np.array([red, blue]) / (.5 * (green_a + green_b)))


def edit_round_trip(path, record, delta, folder, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError('Changed original RGB JPEG')
    with Image.open(path) as image:
        gains = capture_gains(image)
        original, _ = to_srgb(image)
        exif = image.info.get('exif', b'')
    if gains is None:
        raise ValueError('Frozen source lacks readable capture data')
    scale = original.width / 768
    x0, y0, x1, y1 = (round(v * scale) for v in (245, 128, 525, 418))
    crop = np.array(original.crop((x0, y0, x1, y1)))
    eyes = np.array(record['eyes']) * scale - [x0, y0]
    weight = face_weight(crop, eyes)
    altered = local_edit(crop, delta, weight)
    result = original.copy()
    result.paste(Image.fromarray(altered), (x0, y0))
    reports, decoded = {}, {}
    for label, metadata in (('retained', {'exif': exif}), ('removed', {})):
        destination = folder / f'{path.stem}-{label}.jpg'
        result.save(destination, quality=93, subsampling=0, icc_profile=original.info['icc_profile'], **metadata)
        with Image.open(destination) as jpeg:
            reports[label] = dict(fields=capture_gains(jpeg), output_sha256=digest(destination))
            decoded[label] = np.array(jpeg)
    np.testing.assert_array_equal(decoded['retained'], decoded['removed'])
    after = decoded['retained'][y0:y1, x0:x1]
    with BytesIO() as encoded:
        original.save(encoded, format='JPEG', quality=93, subsampling=0, icc_profile=original.info['icc_profile'])
        encoded.seek(0)
        with Image.open(encoded) as jpeg:
            before = np.array(jpeg.crop((x0, y0, x1, y1)))
    face = weight >= .90
    change = float(np.abs(after.astype(np.int16) - before)[face].mean())
    return dict(capture=gains, outputs=reports, face_rgb8_change=change,
                same_decoded_pixels=True, face_pixels=int(face.sum()),
                edited_with_unchanged_gains=change >= 3 and reports['retained']['fields'] == gains,
                stripped_gains_missing=reports['removed']['fields'] is None)


def run(source, transfer, out):
    inputs = checked_json(source / 'report.json', PHOTO_SHA)
    baseline = checked_json(transfer / 'results.json', ARCHIVE)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    if digest(region_file) != baseline['regions_sha256']:
        raise ValueError('Changed face geometry')
    regions = json.loads(region_file.read_bytes())
    out.mkdir()
    cases, shifts = {}, {}
    for group in GROUPS:
        cheek_delta = baseline['admission'][group]['regions']['cheeks']['real']
        for state in ('cold', 'warm'):
            key = f'{group}-{state}'
            cases[key] = edit_round_trip(source / f'{key}.jpg', regions[group][state], np.asarray(cheek_delta),
                                         out, inputs['photos'][key]['sha256'])
        cold, warm = [cases[f'{group}-{state}']['capture'] for state in ('cold', 'warm')]
        gain = normalized(warm['shot_rggb']) - normalized(cold['shot_rggb'])
        change = warm['shot_temperature'] - cold['shot_temperature']
        support = gain[0] < 0 and gain[1] > 0 and np.linalg.norm(gain) > .02 and change < 0
        shifts[group] = dict(log_gain_delta=gain.tolist(), temp_delta=change, face_delta=cheek_delta,
                             qualified=bool(support))
    json_to(out / 'results.json', dict(inputs_sha256=PHOTO_SHA, references_sha256=ARCHIVE,
                                      script_sha256=digest(Path(__file__)), cases=cases, capture_changes=shifts,
                                      capture_cue_qualified=all(row['qualified'] for row in shifts.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'transfer', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
