"""Check if known XMP files differ only in a fixed local brush edit."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

from audit_ppr_metadata import ARCHIVES, ASSETS, NS


CASE = '1645_1'
PUBLISHED_SHA = '47eea57f7dc7d8e75c08ffe67a7d07ac384622bb681db200a4cc432221820e01'
CRS = '{' + NS + '}'
DESCRIPTION = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'
LOCAL = {'PaintBasedCorrections', 'GradientBasedCorrections', 'CircularGradientBasedCorrections',
         'MaskGroupBasedCorrections', 'RetouchAreas', 'RetouchInfo'}
DESCRIPTIVE = {'AlreadyApplied', 'HasSettings', 'Snapshots', 'RawFileName'}
KEY_FIELDS = ('Version', 'ProcessVersion', 'WhiteBalance', 'Temperature', 'Tint', 'Exposure2012',
              'Highlights2012', 'Shadows2012', 'Contrast2012', 'Vibrance', 'Saturation', 'ToneCurvePV2012',
              'CameraProfile', 'CropTop', 'CropLeft', 'CropRight', 'CropBottom', 'Look')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def upstream(role):
    filename, size, expected = ARCHIVES[role]
    path = ASSETS / filename
    if path.stat().st_size != size or sha(path.read_bytes()) != expected:
        raise ValueError('Frozen upstream archive changed')
    with zipfile.ZipFile(path) as archive:
        matches = [info for info in archive.infolist()
                   if PurePosixPath(info.filename).name.lower() == CASE + '.xmp']
        if len(matches) != 1 or not 0 < matches[0].file_size < 128 * 1024:
            raise ValueError('Unexpected upstream XMP member')
        return archive.read(matches[0])


def literal(raw):
    text = ' '.join(raw.split())
    if not text:
        return ''
    if text.casefold() in ('true', 'false'):
        return text.casefold()
    fragments = text.split(',')
    try:
        values = [Decimal(part.strip()) for part in fragments]
    except InvalidOperation:
        return text
    if not all(value.is_finite() for value in values):
        return text
    return ','.join(str(value.normalize()) if value else '0' for value in values)


def subtree(node):
    return (node.tag, tuple(sorted((key, literal(raw)) for key, raw in node.attrib.items())),
            literal(node.text or ''), tuple(subtree(child) for child in node))


def inspect(raw):
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Unsupported XMP declarations')
    xml = ET.fromstring(raw)
    descriptions = [node for node in xml.iter(DESCRIPTION)
                    if CRS + 'ProcessVersion' in node.attrib and CRS + 'WhiteBalance' in node.attrib]
    if len(descriptions) != 1:
        raise ValueError('No unique primary Camera Raw settings')
    node = descriptions[0]
    fields = {name[len(CRS):]: literal(value) for name, value in node.attrib.items() if name.startswith(CRS)}
    ignored = LOCAL | DESCRIPTIVE
    fields = {name: value for name, value in fields.items() if name not in ignored}
    for child in node:
        name = child.tag[len(CRS):]
        if child.tag.startswith(CRS) and name not in ignored:
            if name in fields:
                raise ValueError('Duplicate Camera Raw property encoding')
            fields[name] = subtree(child)
    corrections = {name: sum(entry.get(CRS + 'CorrectionActive', '').casefold() in ('true', '1')
                             for child in node.findall(CRS + name) for entry in child.iter())
                   for name in sorted(LOCAL)}
    return fields, corrections


def compare(left, right):
    shared = left.keys() & right.keys()
    diffs = sorted(name for name in shared if left[name] != right[name])
    return dict(shared=len(shared), differing=len(diffs), left_only=len(left.keys() - right.keys()),
                right_only=len(right.keys() - left.keys()),
                key_field_status={name: ('different' if name in diffs else 'same' if name in shared
                                        else 'missing_on_one_side' if name in left or name in right else 'absent')
                                  for name in KEY_FIELDS})


def run(published, out):
    raw = published.read_bytes()
    if sha(raw) != PUBLISHED_SHA:
        raise ValueError('Frozen MMArt configuration changed')
    documents = {'source': upstream('source'), 'expert_a': upstream('target_a'), 'mmart': raw}
    fields, activity = {}, {}
    for key, payload in documents.items():
        fields[key], activity[key] = inspect(payload)
    assert activity['expert_a']['PaintBasedCorrections'] == 1 and not any(activity['mmart'].values())
    pairs = {f'{a}_vs_{b}': compare(fields[a], fields[b]) for a, b in
             (('source', 'expert_a'), ('source', 'mmart'), ('expert_a', 'mmart'))}
    equivalent = fields['expert_a'] == fields['mmart']
    script = Path(__file__)
    result = dict(sample=CASE, inputs_sha256={key: sha(data) for key, data in documents.items()},
                  active_local=activity, comparisons=pairs, exact_global_match=equivalent,
                  confirmed_toggle=False, media_read=False, source_sha256=sha(script.read_bytes()),
                  protocol_sha256=sha(script.with_name('ppr-brush-toggle-identity-protocol.md').read_bytes()))
    with out.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(pairs['expert_a_vs_mmart']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('published', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.published, args.out)
