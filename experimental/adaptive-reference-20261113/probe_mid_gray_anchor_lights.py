"""Select illumination pairs using a gray probe before measuring wall color."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image


INTAKE_SHA = 'cb7f18370ab588373a1fbcB89d85d0df9f3c1a830c2475f67645867482e8921a'.lower()
DIRECTIONS = (0, 4, 8, 12, 14, 16, 20, 24)
LUMA = np.array([.2126, .7152, .0722])


def read(path):
    return json.loads(path.read_bytes())


def measure(images, box):
    x0, y0, x1, y1 = box
    raw = np.stack([img[y0:y1, x0:x1] for img in images])
    valid = np.all((raw >= 16) & (raw <= 239), axis=(0, 3))
    result = dict(valid=int(valid.sum()), total=valid.size, coverage=float(valid.mean()), qualified=False)
    if not valid.any():
        return result
    rgb = raw[:, valid] / 255.0
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    light = np.median(linear @ LUMA, axis=1)
    logs = np.median(np.log(linear[..., [0, 2]] / linear[..., 1:2]), axis=1)
    delta = logs[1] - logs[0]
    shift = float(np.linalg.norm(delta))
    log_ratio = float(np.log(light[1] / light[0]))
    qualified = result['valid'] >= 128 and result['coverage'] >= .80 and abs(log_ratio) <= .25 and shift >= .06
    return dict(result, luma_ratio=float(np.exp(log_ratio)), log_luma_ratio=log_ratio, delta_chroma=delta.tolist(),
                chroma_magnitude=shift, qualified=qualified)


def run(previous, acquired, ledger_path, out):
    if sha256(ledger_path.read_bytes()).hexdigest() != INTAKE_SHA:
        raise ValueError('Changed previous selection ledger')
    ledger = read(ledger_path)
    additional = read(acquired / 'results.json')
    results = {}
    for scene, archive in ledger['archive_sources'].items():
        new = additional[scene]
        if any(new[field] != archive[field] for field in ('etag', 'archive_bytes', 'index_sha')):
            raise ValueError('Archive version changed')
        combined = archive['members'] | new['members']
        if set(combined) != set(map(str, DIRECTIONS)):
            raise ValueError('Changed direction inventory')
        images = {}
        for direction in DIRECTIONS:
            root = previous if direction in (14, 24) else acquired
            path = root / f'{scene}-{direction}.jpg'
            if sha256(path.read_bytes()).hexdigest() != combined[str(direction)]['sha256']:
                raise ValueError('Changed image')
            with Image.open(path) as image:
                if image.size != (1500, 1000) or image.mode != 'RGB':
                    raise ValueError('Changed image geometry')
                images[direction] = np.array(image)
        boxes = {key: row['box'] for key, row in ledger['material_readings'][scene].items()}
        pairs = []
        for first, second in combinations(DIRECTIONS, 2):
            pair = images[first], images[second]
            pairs.append(dict(directions=[first, second], **measure(pair, boxes['gray_probe'])))
        eligible = [p for p in pairs if p['qualified']]
        best = max(eligible, key=lambda item: item['chroma_magnitude'], default=None)
        walls = {} if best is None else {
            key: measure([images[direction] for direction in best['directions']], box)
            for key, box in boxes.items() if key != 'gray_probe'
        }
        results[scene] = dict(boxes=boxes, members=combined, gray=pairs, selected=best, walls=walls,
                              accepted=any(row['qualified'] for row in walls.values()))
    protocol = Path(__file__).with_name('mid-gray-anchor-lights-protocol.md')
    report = dict(parent=INTAKE_SHA, script_sha=sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha=sha256(protocol.read_bytes()).hexdigest(), scenes=results)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'acquired', 'ledger_path', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
