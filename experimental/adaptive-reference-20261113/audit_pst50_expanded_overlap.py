"""Compare additional consumed photo sources to the sealed PST50 inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from coverage_duplicates import perceptual_hash, pixel_digest
from independent_coco import digest
from probe_pst50_reserve_fingerprints import BASE, ROOT, distance, fingerprint, load_entries, read_json, verify
from triplet_color import to_srgb


def sources():
    paths = {
        'S24': sorted((BASE / 's24-calibrated-pilot-assets-v1').glob('img_*/srgb_images_style_[123].jpg')),
        'INRetouch': sorted((BASE / 'inretouch-public-pilot-v1').glob('*.jpg')),
        'PPR': sorted(
            path for name in ('ppr-pixel-sample-v1', 'ppr-explicit-local-pixels-v1', 'ppr-expert-contrast-v1')
            for path in (BASE / name / 'samples').glob('*/*.tif')
        ),
        'MultiIllum': sorted((BASE / 'multilum-pilot-v1/run').glob('*.jpg')),
    }
    assert {key: len(p) for key, p in paths.items()} == {'S24': 18, 'INRetouch': 8, 'PPR': 22, 'MultiIllum': 4}
    assert sum(path.stat().st_size for items in paths.values() for path in items) <= 96 * 1024**2
    return paths


def tiff_fingerprint(path):
    with Image.open(path) as file:
        srgb, _ = to_srgb(file)
        rgb = np.asarray(srgb)
    h, w = rgb.shape[:2]
    y, x = h // 20, w // 20
    center = rgb[y:h-y, x:w-x]
    return dict(bytes_sha256=digest(path), pixels_sha256=pixel_digest(rgb),
                fingerprints=[perceptual_hash(img) for img in (rgb, rgb[:, ::-1], center, center[:, ::-1])])


def run(out):
    old = BASE / 'pst50-reserve-fingerprint-v1'
    evidence_path = ROOT / 'docs/research/2026-11-18-pst50-reserve-fingerprint-evidence.json'
    evidence = read_json(evidence_path)
    verify(old / 'results.json', evidence['results_sha256'])
    sealed = read_json(old / 'results.json')['fingerprints']
    entries = load_entries()
    assert list(map(int, sealed)) == list(range(16, 51))
    tested = []
    for key, images in sealed.items():
        assert set(images) == {'content_709', 'paired_style'}
        for role, row in images.items():
            entry = entries[role][int(key)]
            verify(old / 'inputs' / entry['path'], entry['lfs']['oid'])
            assert row['bytes_sha256'] == entry['lfs']['oid']
            tested.append((int(key), role, row))
    assert len(tested) == 70

    cohorts = {}
    for group, paths in sources().items():
        rows = []
        for path in paths:
            photo = tiff_fingerprint(path) if group == 'PPR' else fingerprint(path)
            distances, matches = [], []
            for index, role, reserve in tested:
                gap = distance(photo['fingerprints'], reserve['fingerprints'])
                same_bytes = photo['bytes_sha256'] == reserve['bytes_sha256']
                same_pixels = photo['pixels_sha256'] == reserve['pixels_sha256']
                distances.append(gap)
                if gap <= 6 or same_bytes or same_pixels:
                    matches.append(dict(index=index, role=role, gap=gap, same_bytes=same_bytes, same_pixels=same_pixels))
            rows.append(dict(file=path.relative_to(BASE).as_posix(), bytes=path.stat().st_size,
                             sha256=photo['bytes_sha256'], closest=min(distances), matches=matches))
        cohorts[group] = rows
    direct = {match['index'] for items in cohorts.values() for row in items for match in row['matches']}
    isolated = sorted(direct.union(*(set(component) for component in evidence['components']
                                    if direct.intersection(component))))
    method = Path(__file__)
    report = dict(reserved_inputs=len(tested), old_results_sha256=evidence['results_sha256'],
                  groups=cohorts, additional_isolated=isolated, source_sha256=digest(method),
                  protocol_sha256=digest(method.with_name('pst50-expanded-overlap-protocol.md')))
    out.mkdir()
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: [len(rows), min(row['closest'] for row in rows)] for key, rows in cohorts.items()}))
    print(json.dumps(dict(additional_isolated=isolated)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
