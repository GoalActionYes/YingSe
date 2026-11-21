"""Audit declared color fields in the original, checksum-verified JPEG files."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from probe_target_mirror_chroma import BASE, ROOT, digest, json_to

STANDARD = b'http://ns.adobe.com/xap/1.0/' + bytes(1)
EXTENDED = b'http://ns.adobe.com/xmp/extension/' + bytes(1)
CRS = '{http://ns.adobe.com/camera-raw-settings/1.0/}'
FIELDS = {'Exposure', 'Exposure2012', 'Contrast', 'Contrast2012', 'Brightness', 'Saturation',
          'Vibrance', 'Temperature', 'Tint', 'ShadowTint', 'ToneCurve', 'ToneCurvePV2012',
          'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012',
          'RedHue', 'GreenHue', 'BlueHue', 'RedSaturation', 'GreenSaturation', 'BlueSaturation'}


def color_fields(root):
    selected = {}
    for node in root.iter():
        for key, value in node.attrib.items():
            name = key.removeprefix(CRS)
            if key.startswith(CRS) and name in FIELDS and value.strip():
                selected[name] = value.strip()
        name = node.tag.removeprefix(CRS)
        if node.tag.startswith(CRS) and name in FIELDS:
            values = [value.strip() for value in node.itertext() if value.strip()]
            if values:
                selected[name] = tuple(values)
    return selected


def read_image(source):
    with Image.open(source) as image:
        if image.format != 'JPEG':
            raise ValueError('Expected original JPEG export')
        segments = [raw for label, raw in image.applist if label == 'APP1']
        standard = [raw[len(STANDARD):] for raw in segments if raw.startswith(STANDARD)]
        extended = sum(raw.startswith(EXTENDED) for raw in segments)
        values, errors = {}, 0
        for packet in standard:
            try:
                values.update(color_fields(ElementTree.fromstring(packet.rstrip(bytes(1)))))
            except ElementTree.ParseError:
                errors += 1
        exif = image.getexif()
        return dict(packets=len(standard), extended=extended, unreadable=errors, exif=bool(exif),
                    icc=bool(image.info.get('icc_profile')), software=bool(exif.get(305)),
                    field_names=sorted(values)), values


def check_parser():
    packet = ('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
              'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
              '<rdf:Description crs:Exposure2012="0.4"><crs:Saturation>-30</crs:Saturation>'
              '</rdf:Description></rdf:RDF>').encode()
    file = BytesIO()
    Image.new('RGB', (12, 12), (90, 112, 125)).save(file, format='JPEG', xmp=packet)
    file.seek(0)
    tags, values = read_image(file)
    if tags['packets'] != 1 or values != {'Exposure2012': '0.4', 'Saturation': ('-30',)}:
        raise ValueError('Missing tagged positive control')


def verified(path, receipt, read):
    if digest(path) != receipt['sha256']:
        raise ValueError(f'Changed frozen JPEG: {path.name}')
    return read(path)


def pairs(*, read=read_image):
    evidence = ROOT / 'docs/research'
    facd = json.loads((evidence / '2026-11-18-facd-range-pilot-evidence.json').read_bytes())['cases']
    for group, records in facd.items():
        directory = BASE / 'facd-range-media-v1' / group
        yield 'facd', group, {label: verified(directory / f'{label}.jpg', record, read)
                              for label, record in records.items()}
    directory = BASE / 'psbench-background-desaturation-assets-v1'
    archived = BASE / 'psbench-background-desaturation-v1/results.json'
    receipts = json.loads(archived.read_bytes())['input_files']
    for group in ('145', '248'):
        yield 'psbench', group, {role: verified(directory / f'{group}_{role}.jpg', receipts[f'{group}_{role}.jpg'], read)
                                 for role in ('start', 'end')}
    directory = BASE / 'psbench-local-intent-pair-v1'
    known = evidence / '2026-11-19-psbench-local-intent-pair-evidence.json'
    for record in json.loads(known.read_bytes())['source_checks']:
        group = str(record['task'])
        yield 'psbench', group, {role: verified(directory / f'{group}_{role}.jpg', item, read)
                                 for role, item in zip(('start', 'end'), record['files'], strict=True)}


def run(out):
    check_parser()
    rows, coverage, totals = {}, Counter(facd=0, psbench=0), Counter()
    file_count = 0
    for domain, name, inputs in pairs():
        origin = 'Origin' if domain == 'facd' else 'start'
        _, original = inputs[origin]
        frames, differences, edited_signatures = {}, {}, set()
        for label, (note, fields) in inputs.items():
            frames[label] = note
            file_count += 1
            if label != origin:
                differences[label] = sorted(key for key in fields
                                            if original.get(key) != fields.get(key))
                edited_signatures.add(tuple(sorted(fields.items())))
                totals[domain] += 1
                coverage[domain] += int(bool(differences[label]))
        rows[f'{domain}-{name}'] = dict(frames=frames, changed_fields=differences,
                                      distinct_edited_records=len(edited_signatures))
    if file_count != 40 or totals != {'facd': 24, 'psbench': 4}:
        raise ValueError('Changed frozen example counts')
    out.mkdir()
    json_to(out / 'results.json', dict(file_count=file_count, candidates=coverage, edited_totals=totals,
                                      cases=rows, passed=coverage['facd'] >= 18 and coverage['psbench'] >= 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
