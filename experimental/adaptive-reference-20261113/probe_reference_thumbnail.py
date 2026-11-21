"""Check whether exported JPEG thumbnails preserve the ungraded reference."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from struct import pack

import numpy as np
from PIL import ExifTags, Image, ImageCms, UnidentifiedImageError

from probe_reference_embedded_edit import pairs
from probe_target_mirror_chroma import json_to


SRGB = ImageCms.createProfile('sRGB')


def to_rgb(image, orientation):
    operations = {
        2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    profile = image.info.get('icc_profile')
    frame = (ImageCms.profileToProfile(image, BytesIO(profile), SRGB, outputMode='RGB')
             if profile else image.convert('RGB'))
    return frame.transpose(operations[orientation]) if orientation in operations else frame


def read(source):
    with Image.open(source) as image:
        if image.format != 'JPEG':
            raise ValueError('Expected JPEG')
        exif = image.getexif()
        base = to_rgb(image, exif.get(274, 1))
        ifd = exif.get_ifd(ExifTags.IFD.IFD1)
        start, length = ifd.get(513), ifd.get(514)
        if start is None and length is None:
            return base, None, 'absent'
        packet = image.info.get('exif', b'')
        if packet.startswith(b'Exif\x00\x00'):
            packet = packet[6:]
        if (not isinstance(start, int) or not isinstance(length, int)
                or start < 8 or length < 4 or start + length > len(packet)):
            return base, None, 'invalid_offset'
        data = packet[start:start+length]
        try:
            with Image.open(BytesIO(data)) as preview:
                if preview.format != 'JPEG':
                    return base, None, 'not_jpeg'
                thumbnail = to_rgb(preview, ifd.get(274, exif.get(274, 1)))
        except (UnidentifiedImageError, OSError, ImageCms.PyCMSError):
            return base, None, 'decode_failed'
        return base, thumbnail, 'decoded'


def classify(original, current, thumbnail):
    if min(thumbnail.size) < 48:
        return {'classification': 'too_small'}
    thumb_aspect = thumbnail.width / thumbnail.height
    if any(abs(frame.width / frame.height / thumb_aspect - 1) > .05
           for frame in (original, current)):
        return {'classification': 'aspect_mismatch'}
    views = [np.asarray(frame.resize(thumbnail.size, Image.Resampling.BOX), dtype=np.int16)
             for frame in (original, current, thumbnail)]
    before, after, thumb = views
    before_error = float(np.abs(before - thumb).mean())
    after_error = float(np.abs(after - thumb).mean())
    edit = float(np.abs(before - after).mean())
    result = 'ambiguous'
    if edit >= 6:
        if before_error <= 6 and after_error - before_error >= 3:
            result = 'before'
        if after_error <= 6 and before_error - after_error >= 3:
            result = 'current'
    return {'classification': result, 'before_error': before_error,
            'current_error': after_error, 'edit_difference': edit}


def fixture(main, preview):
    file = BytesIO()
    options = {}
    if preview is not None:
        encoded = BytesIO()
        preview.save(encoded, format='JPEG', quality=95)
        jpg = encoded.getvalue()
        # TIFF header, empty IFD0, IFD1 with the two LONG JPEG tags, then bytes.
        start = 8 + 6 + 2 + 24 + 4
        exif = (b'Exif\x00\x00II' + pack('<HIHIH', 42, 8, 0, 14, 2)
                + pack('<HHIIHHIII', 513, 4, 1, start, 514, 4, 1, len(jpg), 0) + jpg)
        options['exif'] = exif
    main.save(file, format='JPEG', quality=95, **options)
    file.seek(0)
    return file


def check_fixture():
    a = Image.new('RGB', (128, 96), (70, 110, 125))
    b = Image.new('RGB', (128, 96), (125, 105, 90))
    for preview, expected in [(None, 'absent'), (a, 'before'), (b, 'current')]:
        thumbnail_source = preview.resize((64, 48)) if preview is not None else None
        full, thumb, state = read(fixture(b, thumbnail_source))
        actual = classify(a, full, thumb)['classification'] if thumb is not None else state
        if actual != expected:
            raise ValueError(f'Fixture expected {expected}, got {actual}')


def run(out):
    check_fixture()
    rows = {}
    eligible_groups = []
    count = 0
    for source, group, frames in pairs(read=read):
        before_key = 'Origin' if source == 'facd' else 'start'
        original = frames[before_key][0]
        row = {}
        for key, (frame, thumbnail, state) in frames.items():
            count += 1
            detail = {'status': state, 'size': list(thumbnail.size) if thumbnail is not None else None}
            if thumbnail is not None and key != before_key:
                detail.update(classify(original, frame, thumbnail))
            row[key] = detail
        identity = f'{source}-{group}'
        if any(item.get('classification') == 'before' for item in row.values()):
            eligible_groups.append(identity)
        rows[identity] = row
    if count != 40:
        raise ValueError('Expected 40 frozen JPEGs')
    out.mkdir()
    json_to(out / 'results.json', {'files': count, 'groups': rows, 'eligible_groups': eligible_groups,
                                   'advance': len(eligible_groups) >= 2})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
