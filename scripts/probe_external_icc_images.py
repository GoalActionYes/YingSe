"""Check original third-party ICC JPEGs in both stable image input roles."""

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageCms

from probe_reference_icc_video import png, references, request, sha
from reference_color_app.server import make_server


INPUTS = {
    'Italy-P3.jpg': ('8e78d38839fa044bbcc0dec99c91c82561205bc691d00c4128528d89459bb7ed', 'Display P3'),
    'YellowFlower-P3.jpg': ('59a31408f7997a79ad57e67dd9ee64aa1719f3aa5ebe2d51304e2c2cdbbff59f', 'Display P3'),
    'Shoes-AdobeRGB.jpg': ('1dd8f791a9c769c0c72554bee7aa683e1dcd84f86d93d9decc4da57ff6fb4c8c', 'Adobe RGB (1998)'),
}


def pixels(raw):
    with Image.open(BytesIO(raw)) as image:
        return np.array(image, dtype=np.int16)


def from_url(text):
    return base64.b64decode(text.split(',', 1)[1])


def counterpart():
    y, x = np.indices((64, 96))
    return png(Image.fromarray(np.stack((24 + 2*x, 48 + 2*y, 80 + x//2 + y), axis=-1).astype(np.uint8)))


def verify_input(path, digest, description):
    if sha(path.read_bytes()) != digest:
        raise ValueError(f'Changed source photo: {path.name}')
    with Image.open(path) as image:
        if image.format != 'JPEG' or image.size != (600, 400) or image.getexif().get(274) != 1:
            raise ValueError(f'Unexpected JPEG geometry or orientation: {path.name}')
        icc = image.info['icc_profile']
        embedded = ImageCms.ImageCmsProfile(BytesIO(icc))
        if ImageCms.getProfileDescription(embedded).strip() != description:
            raise ValueError(f'Unexpected ICC: {path.name}')
    return sha(icc)


def render(url, target, reference):
    mapping = request(url, '/api/analyze', {'target_id': target, 'reference_id': reference})
    params = {'mapping_id': mapping['mapping_id'], 'strength': .8,
              'adjustments': {'basic': {'brightness': 12}, 'hsl': {'red': [0, 8, 0]}}}
    preview = request(url, '/api/preview', params)
    job = request(url, '/api/export', dict(params, media_id=target, kind='png'))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = request(url, '/api/jobs/' + job['id'])
        if state['status'] == 'done':
            result = request(url, state['result_url'])
            np.testing.assert_array_equal(pixels(result), pixels(from_url(preview['after'])))
            return result
        if state['status'] not in ('queued', 'running'):
            raise RuntimeError(state.get('error') or state['status'])
        time.sleep(.05)
    raise TimeoutError('PNG export did not finish')


def check_photo(url, control, path, out):
    digest, description = INPUTS[path.name]
    profile = verify_input(path, digest, description)
    controls = references(path)
    reference = pixels(controls['srgb'])
    original = pixels(controls['untagged'])
    if np.abs(reference - original).mean() <= 1:
        raise ValueError('ICC control has insufficient signal')
    cases = {}
    for arm, data in controls.items():
        extension = 'jpg' if arm == 'tagged' else 'png'
        ident = request(url, '/api/media', data, f'{path.stem}-{arm}.{extension}')['id']
        before = pixels(from_url(request(url, '/api/frame', {'media_id': ident})['image']))
        np.testing.assert_array_equal(before, original if arm == 'untagged' else reference)
        cases[arm] = {'target': render(url, ident, control), 'reference': render(url, control, ident)}
    report = {}
    for role in ('target', 'reference'):
        images = {arm: values[role] for arm, values in cases.items()}
        expected = pixels(images['srgb'])
        errors = {arm: float(np.abs(pixels(raw) - expected).mean()) for arm, raw in images.items()}
        report[role] = {'mae': errors, 'sha256': {arm: sha(raw) for arm, raw in images.items()},
                        'passed': errors['tagged'] == 0 and errors['untagged'] >= .5}
        for arm, raw in images.items():
            (out / f'{path.stem}-{role}-{arm}.png').write_bytes(raw)
    return {'source_sha256': digest, 'icc': description, 'icc_sha256': profile,
            'decoded_untagged_mae': float(np.abs(reference-original).mean()), 'roles': report}


def run(inputs, out):
    out.mkdir()
    server = make_server(out / 'store')
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        control = request(url, '/api/media', counterpart(), 'counterpart.png')['id']
        rows = {name: check_photo(url, control, inputs / name, out) for name in INPUTS}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    passed = all(role['passed'] for row in rows.values() for role in row['roles'].values())
    (out / 'results.json').write_text(json.dumps({'photos': rows, 'passed': passed}, indent=2) + '\n',
                                      encoding='utf-8', newline='\n')
    if not passed:
        raise SystemExit('A JPEG input role did not meet the parity and control gates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
