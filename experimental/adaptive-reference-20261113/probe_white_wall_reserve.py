"""Qualify fixed white-wall photographs without evaluating any color models."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_pst50_reserve_fingerprints import cached_consumed, distance, fingerprint, load_entries, local_consumed
from probe_reserve_recent_source_crosscheck import consumed, reserved
from reference_color_app.media import decode_image

FILES = ('hallway.jpg', 'living-room.jpg', 'apartment.jpg', 'office.jpg')


def compare(first, second):
    gap = distance(first['fingerprints'], second['fingerprints'])
    duplicate = (first['bytes_sha256'] == second['bytes_sha256']
                 or first['pixels_sha256'] == second['pixels_sha256']
                 or gap <= 6)
    return gap, duplicate


def reference_groups():
    latest = {}
    for entries in consumed().values():
        for path, expected in entries:
            stamp = fingerprint(path)
            if stamp['bytes_sha256'] != expected:
                raise ValueError(f'Changed consumed image {path.name}')
            latest[path.name] = stamp
    return {
        'older_consumed': local_consumed(load_entries()),
        'recent_consumed': latest,
        'expansion_training': {label: {'bytes_sha256': key, 'pixels_sha256': pixels,
                                       'fingerprints': [value]}
                               for key, (label, value, pixels) in cached_consumed().items()},
        **reserved(),
    }


def qualify_region(path, box):
    rgb = decode_image(path)
    image = Image.fromarray(rgb)
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError('Wall rectangle outside input')
    sample = np.asarray(image)[y0:y1, x0:x1]
    # Assess measurable headroom only. Region registration identifies the wall.
    valid = np.all((sample >= 24) & (sample <= 239), axis=2)
    return dict(analysis_size=list(image.size), box=box, usable_pixels=int(valid.sum()),
                median_rgb=np.median(sample[valid], axis=0).tolist() if valid.any() else None,
                qualified=bool(valid.sum() >= 256))


def run(folder, manifest_path, out):
    manifest = json.loads(manifest_path.read_bytes())
    if tuple(manifest) != FILES:
        raise ValueError('Changed frozen photo order')
    pool = reference_groups()
    images, stamps = {}, {}
    for name, row in manifest.items():
        if row.get('missing'):
            images[name] = {'missing': row['missing'], 'qualified': False}
            continue
        path = folder / name
        stamp = fingerprint(path)
        if stamp['bytes_sha256'] != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError(f'Changed archived file {name}')
        stamps[name] = stamp
        measurements = qualify_region(path, row['box'])
        images[name] = {'sha256': stamp['bytes_sha256'], 'pixels_sha256': stamp['pixels_sha256'],
                        'pairing': {}, **measurements}
        for group, prior in pool.items():
            pairs = [(label, *compare(stamp, second)) for label, second in prior.items()]
            alerts = [label for label, _, flagged in pairs if flagged]
            images[name]['pairing'][group] = dict(count=len(pairs), min_distance=min(gap for _, gap, _ in pairs),
                                                alerts=alerts)
            if alerts:
                images[name]['qualified'] = False
    cross = []
    for (first, source), (second, target) in combinations(stamps.items(), 2):
        gap, flagged = compare(source, target)
        cross.append({'pair': [first, second], 'distance': gap, 'flagged': flagged})
        if flagged:
            images[first]['qualified'] = images[second]['qualified'] = False
    payload = dict(images=images, within=cross, qualified=sum(row['qualified'] for row in images.values()),
                   models_run=0)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('folder', 'manifest_path', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
