"""Verify a tagged CMYK JPEG through both workbenches' real HTTP paths."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.error import HTTPError

import numpy as np
from PIL import Image, ImageCms, ImageOps

from probe_external_icc_images import counterpart, from_url, pixels, render
from probe_reference_icc_video import png, request, sha
from reference_color_app.server import make_server
from spatial_workbench import make_spatial_workbench


def fixtures(source, profile_bytes):
    profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
    srgb = ImageCms.createProfile('sRGB')
    intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
    with Image.open(source) as image:
        image.thumbnail((384, 256), Image.Resampling.BILINEAR)
        cmyk = ImageCms.profileToProfile(image.convert('RGB'), srgb, profile, outputMode='CMYK', renderingIntent=intent)
    exif = Image.Exif()
    exif[274] = 6
    encoded = BytesIO()
    cmyk.save(encoded, 'JPEG', quality=95, icc_profile=profile_bytes, exif=exif)
    tagged = encoded.getvalue()
    with Image.open(BytesIO(tagged)) as jpeg:
        oriented = ImageOps.exif_transpose(jpeg)
        expected = ImageCms.profileToProfile(oriented, profile, srgb, outputMode='RGB', renderingIntent=intent)
        untagged = BytesIO()
        Image.frombytes('CMYK', oriented.size, oriented.tobytes()).save(untagged, 'TIFF')
    with Image.open(BytesIO(untagged.getvalue())) as control:
        assert not control.info.get('icc_profile')
    # The sRGB pixels are already normalized. Drop the generated profile's timestamp.
    srgb_pixels = Image.frombytes('RGB', expected.size, expected.tobytes())
    return {'tagged.jpg': tagged, 'srgb.png': png(srgb_pixels), 'untagged.tif': untagged.getvalue()}


def check(server, inputs, output):
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        ids, decoded = {}, {}
        for name, raw in inputs.items():
            ids[name] = request(url, '/api/media', raw, name)['id']
            frame = request(url, '/api/frame', {'media_id': ids[name]})
            decoded[name] = pixels(from_url(frame['image']))
        control = request(url, '/api/media', counterpart(), 'counterpart.png')['id']
        errors, hashes = {}, {}
        for role in ('target', 'reference'):
            results = {name: render(url, *((ident, control) if role == 'target' else (control, ident)))
                       for name, ident in ids.items()}
            expected = pixels(results['srgb.png'])
            errors[role] = {name: float(np.abs(pixels(raw) - expected).mean()) for name, raw in results.items()}
            hashes[role] = {}
            for name, raw in results.items():
                (output / f'{role}-{Path(name).stem}.png').write_bytes(raw)
                hashes[role][name] = sha(raw)
        differences = {name: float(np.abs(rgb - decoded['srgb.png']).mean()) for name, rgb in decoded.items()}
        passed = differences['tagged.jpg'] == 0 and differences['untagged.tif'] >= 1
        passed &= all(row['tagged.jpg'] == 0 and row['untagged.tif'] >= .5 for row in errors.values())
        return {'decode_mae': differences, 'export_mae': errors, 'output_sha256': hashes, 'passed': passed}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def run(source, profile, out):
    out.mkdir()
    profile_bytes = profile.read_bytes()
    inputs = fixtures(source, profile_bytes)
    rows = {}
    for name, factory in (('stable', make_server), ('spatial', lambda data: make_spatial_workbench(data, enhanced=True))):
        local = out / name
        local.mkdir()
        try:
            rows[name] = check(factory(local / 'store'), inputs, local)
        except HTTPError as error:
            rows[name] = {'passed': False, 'http_status': error.code, 'detail': error.read().decode('utf-8')}
    report = {'source_sha256': sha(source.read_bytes()), 'profile_sha256': sha(profile_bytes),
              'fixtures_sha256': {name: sha(data) for name, data in inputs.items()}, 'workbenches': rows}
    for name, data in inputs.items():
        (out / name).write_bytes(data)
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False))
    if not all(row['passed'] for row in rows.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'profile', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
