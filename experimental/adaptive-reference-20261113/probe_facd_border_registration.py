"""Compare fixed crop and interior-rescale hypotheses for FACD's black frames."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_facd_range import FILTERS
from probe_tst_reference_spatial_signal import structure_score
from reference_color_app.media import decode_image


def leading_frame(lines):
    detected = lines.mean(axis=1) >= .98
    stops = np.flatnonzero(~detected)
    length = int(stops[0]) if len(stops) else len(lines)
    return length if 2 <= length < len(lines) else 0


def frame(rgb):
    dark = np.all(rgb <= 12, axis=2)
    h, w = dark.shape
    vertical = dark[:, w // 10:w - w // 10]
    horizontal = dark[h // 10:h - h // 10].T
    h_scan, w_scan = h // 20, w // 20
    left, right = (leading_frame(x[:w_scan]) for x in (horizontal, horizontal[::-1]))
    top, bottom = (leading_frame(x[:h_scan]) for x in (vertical, vertical[::-1]))
    return left, top, right, bottom


def box(width, height, edges):
    left, top, right, bottom = edges
    return left, top, width - right, height - bottom


def compare(before, after):
    resized = [np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR)) for image in (before, after)]
    return structure_score(*resized)


def run(source, evidence, out):
    parent = json.loads(evidence.read_text(encoding='utf-8'))
    if len(parent['selected']) != 8 or parent['audited_archive_members'] != 32:
        raise ValueError('Expected the frozen FACD pilot')
    out.mkdir(exist_ok=False)
    cases = {}
    for group in parent['selected']:
        variants = {}
        for kind in FILTERS:
            path = source / group / f'{kind}.jpg'
            if digest(path) != parent['cases'][group][kind]['sha256']:
                raise ValueError(f'Image identity changed: {group}/{kind}')
            variants[kind] = decode_image(path)
        origin = variants.pop('Origin')
        if any(rgb.shape != origin.shape for rgb in variants.values()):
            raise ValueError(f'Image geometry changed: {group}')
        before = Image.fromarray(origin)
        edges = {kind: frame(rgb) for kind, rgb in variants.items()}
        common_edges = (np.max(list(edges.values()), axis=0) + 4).tolist()
        common = box(*before.size, common_edges)
        interior = before.crop(common)
        folder = out / group
        folder.mkdir()
        interior.save(folder / 'Origin.png')
        checks = {}
        hashes = {'Origin.png': digest(folder / 'Origin.png')}
        for kind, rgb in variants.items():
            after = Image.fromarray(rgb)
            raw = compare(before, after)
            prior = parent['cases'][group][kind]['geometry']
            if not (raw['passed'] == prior['passed'] and
                    (raw['whole']['dy'], raw['whole']['dx']) == (prior['whole']['dy'], prior['whole']['dx']) and
                    abs(raw['whole']['zero'] - prior['whole']['zero']) < 1e-10 and
                    abs(raw['maximum_quadrant_shift_gain'] - prior['maximum_quadrant_shift_gain']) < 1e-10):
                raise ValueError(f'Raw alignment scores changed: {group}/{kind}')
            corrected = dict(overlay=after.crop(common), rescale=after.crop(
                box(*before.size, edges[kind])).resize(before.size, Image.Resampling.BILINEAR).crop(common))
            checks[kind] = dict(original=raw, **{method: compare(interior, value)
                                                 for method, value in corrected.items()})
            for method, image in corrected.items():
                path = folder / f'{kind}-{method}.png'
                image.save(path)
                hashes[path.name] = digest(path)
        cases[group] = dict(measured_border_ltrb=edges, common_crop=common, checks=checks, artifacts=hashes)
    methods = ('original', 'overlay', 'rescale')
    counts = {method: sum(all(row[method]['passed'] for row in case['checks'].values())
                          for case in cases.values()) for method in methods}
    pair_counts = {method: sum(row[method]['passed'] for case in cases.values() for row in case['checks'].values())
                   for method in methods}
    summary = dict(complete_families=counts, passed_pairs=pair_counts,
                   eligible={method: counts[method] >= 6 for method in methods}, cases=cases,
                   evidence_sha256=digest(evidence), script_sha256=digest(__file__),
                   protocol_sha256=digest(Path(__file__).with_name('facd-border-registration-protocol.md')))
    (out / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete_families=counts, passed_pairs=pair_counts)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'evidence', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
