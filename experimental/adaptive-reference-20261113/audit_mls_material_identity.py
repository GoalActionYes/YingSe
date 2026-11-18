"""Join MLS preview mask labels to their released surface spectra."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


FILES = {
    'masks_preview.zip': (63177, '27aa88829628552d8bf3e03e687c04776a31b85f89dfcbd9df26d9b78d8cacae'),
    'surfaces.zip': (910419, '4d464686b286faade8efc1eed8777505aa789b7e0d926d7ad471091f0fa2bb00'),
}


def checked(folder, name):
    path = folder/name
    expected_size, expected_hash = FILES[name]
    assert path.stat().st_size == expected_size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    return ZipFile(path)


def table(archive, name):
    member = archive.getinfo(name)
    assert member.file_size <= 100000
    return list(csv.DictReader(io.StringIO(archive.read(member).decode('utf-8-sig'))))


def valid_spectrum(rows):
    values = np.array([(float(row['wavelength']), float(row['value'])) for row in rows])
    return bool(len(values) > 100 and np.isfinite(values).all()
                and (np.diff(values[:, 0]) > 0).all() and values[0, 0] <= 381
                and values[-1, 0] >= 779 and (values[:, 1] > 0).any())


def run(folder, out):
    labels, counts = defaultdict(dict), defaultdict(dict)
    missing_masks, extra_colors, spectra = [], [], {}
    with checked(folder, 'surfaces.zip') as surface, checked(folder, 'masks_preview.zip') as masks:
        names = set(surface.namelist())
        rows = table(surface, 'surfaces/mask_to_surface.csv')
        frequencies = Counter((int(row['scene']), tuple(map(int, row['mask_colour_rgb'].split()))) for row in rows)
        rejected = []
        for row in rows:
            scene = int(row['scene'])
            color = tuple(map(int, row['mask_colour_rgb'].split()))
            if not (1 <= scene <= 24 and len(color) == 3 and all(0 <= n <= 255 for n in color)):
                rejected.append(dict(**row, reason='out_of_range'))
                continue
            if frequencies[scene, color] != 1:
                rejected.append(dict(**row, reason='duplicate_color'))
                continue
            labels[scene][color] = row['surface']
        assert set(labels) == set(range(1, 25))
        for name in sorted({name for by_scene in labels.values() for name in by_scene.values()}):
            member = f'surfaces/spectra/{name}.csv'
            spectra[name] = ('missing' if member not in names else
                             'valid' if valid_spectrum(table(surface, member)) else 'invalid')
        for scene, by_color in sorted(labels.items()):
            member = masks.getinfo(f'masks_preview/{scene:02d}.png')
            assert member.file_size <= 500000
            with Image.open(io.BytesIO(masks.read(member))) as image:
                assert image.mode == 'RGB' and image.size == (578, 475)
                used = {color: amount for amount, color in image.getcolors(maxcolors=578*475)}
            for color, material in by_color.items():
                amount = used.get(color, 0)
                if not amount:
                    missing_masks.append(dict(scene=scene, material=material, mask_rgb=color))
                counts[material][scene] = counts[material].get(scene, 0) + amount
            for color in sorted(set(used)-set(by_color)):
                extra_colors.append(dict(scene=scene, mask_rgb=color, pixels=used[color]))
    eligible = {surface: sorted(scene for scene, pixels in region.items() if pixels >= 256)
                for surface, region in sorted(counts.items()) if spectra[surface] == 'valid'}
    repeated = {surface: scenes for surface, scenes in eligible.items() if len(scenes) >= 2}
    panels = []
    for a, b in combinations(repeated, 2):
        common = sorted(set(repeated[a]).intersection(repeated[b]))
        if len(common) >= 2:
            panels.append(dict(surfaces=[a, b], scenes=common))
    result = dict(archives=FILES, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scene_count=len(labels), mapping_rows=len(rows), rejected_rows=rejected,
                  spectra_summary=Counter(spectra.values()), spectral_status=spectra,
                  pixels_by_surface=counts, absent_mask_entries=missing_masks, unmapped_colors=extra_colors,
                  repeated_materials=repeated, crossed_panels=panels, proceeded=bool(panels))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(spectra=result['spectra_summary'], repeated=len(repeated), panels=len(panels),
                          absent=len(missing_masks), unmapped=len(extra_colors))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.folder, args.out)
