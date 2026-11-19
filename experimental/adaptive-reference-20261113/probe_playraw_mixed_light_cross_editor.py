"""Compare frozen chroma replay to an independent editor on a different photograph."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from audit_playraw_sky_xmp import DT, RDF, payload
from probe_playraw_mixed_light_skin import INPUT_HASHES, export, read
from probe_playraw_mixed_light_transfer import LUMA, STRENGTH, chroma, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, json_to, save
from spatial_render import finish_spatial


FIT_SHA = '1176c40bd8d73c213250df803aee7d148e982625db4cb9fef5a205e82213b054'
INPUTS = {
    'raw': ('rawtherapee-portrait-toning-inputs-v1/DSC00178.ARW',
            '7fbb1b6b9509e3ebb81aa9e76354c09ff1bf3002f465ef0364fcab6f8595304d'),
    'target': ('darktable-cuckoo-local-v1/off.xmp',
               'ef95b1097a27ae604c82914a9ae2419fe9321fec837b6fb5138ab95c954686a8'),
    'legacy': ('darktable-cuckoo-legacy-v1/off.png',
               '4104a99302814307ad560dfa7d2e9c1a6c2860af13d16db3a3d843f0880b0da0'),
    'author': ('playraw-mixed-light-skin-v1/author.xmp', INPUT_HASHES['author.xmp']),
    'cli': ('playraw-mixed-light-skin-v1/runtime/bin/darktable-cli.exe', INPUT_HASHES['darktable-cli.exe']),
}
REGIONS_SHA = 'ef1fa9dac54d2fb706134171aef540bcd045e977d08939fd49dff27ac14513a5'
DEFAULT_PIXELS = '0b59e9ce48841d10676a1b69767bddf22b512a416214efcb86e214591c967f95'
SKIN = ('cheek', 'chest')
PROTECTED = ('light_material', 'green_background')


def step(xmp, number):
    row, = [node for node in ET.fromstring(xmp).iter(RDF + 'li') if node.get(DT + 'num') == number]
    if row.get(DT + 'operation') != 'colorbalancergb' or row.get(DT + 'blendop_version') != '11':
        raise ValueError('Unexpected color balance step')
    return row.attrib


def states(target, source):
    color = step(source, '28')
    before = step(target, '10')
    blend = payload(color[DT + 'blendop_params'])
    if (before[DT + 'enabled'], color[DT + 'enabled'], color[DT + 'modversion'], blend[:4]) != (
            '0', '1', '5', b'\x03\x00\x00\x00'):
        raise ValueError('Unexpected operation state or mask')
    updates = {'modversion': color[DT + 'modversion'], 'params': color[DT + 'params'],
               'blendop_params': (b'\x01\x00\x00\x00' + blend[4:]).hex()}
    row, = [tag for tag in re.finditer(rb'<rdf:li\s+[^>]*>', target) if b'darktable:num="10"' in tag[0]]
    tag = row[0]
    for name, value in updates.items():
        attribute = f'darktable:{name}="'.encode()
        tag, count = re.subn(attribute + rb'[^"]*"', lambda _: attribute + value.encode() + b'"', tag)
        if count != 1:
            raise ValueError(f'Unexpected attribute count: {name}')
    off = target[:row.start()] + tag + target[row.end():]
    on = target[:row.start()] + tag.replace(b'darktable:enabled="0"', b'darktable:enabled="1"') + target[row.end():]
    expected = before | {DT + k: v for k, v in updates.items()}
    if step(off, '10') != expected or step(on, '10') != expected | {DT + 'enabled': '1'}:
        raise ValueError('Unexpected edit-state difference')
    if sum(a != b for a, b in zip(off, on, strict=True)) != 1:
        raise ValueError('More than one byte changed between the two states')
    return off, on


def default_states(source):
    color = step(source, '28')
    blend = payload(color[DT + 'blendop_params'])
    if (color[DT + 'enabled'], color[DT + 'modversion'], blend[:4]) != ('1', '5', b'\x03\x00\x00\x00'):
        raise ValueError('Unexpected source operation state')
    attrs = {'num': '0', 'operation': 'colorbalancergb', 'modversion': '5', 'enabled': '0',
             'params': color[DT + 'params'], 'blendop_version': '11',
             'blendop_params': (b'\x01\x00\x00\x00' + blend[4:]).hex(), 'multi_priority': '0', 'multi_name': ''}
    item = ' '.join(f'darktable:{key}="{value}"' for key, value in attrs.items())
    off = f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:darktable="http://darktable.sf.net/"
    darktable:xmp_version="5" darktable:history_end="1" darktable:iop_order_version="2">
   <darktable:history><rdf:Seq><rdf:li {item}/></rdf:Seq></darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>'''.encode()
    on = off.replace(b'darktable:enabled="0"', b'darktable:enabled="1"')
    if step(off, '0') != {DT + key: value for key, value in attrs.items()} or sum(
            a != b for a, b in zip(off, on, strict=True)) != 1:
        raise ValueError('Unexpected default sidecar')
    return off, on


def alignment(before, legacy):
    if before.shape != legacy.shape or before.shape != (683, 1024, 3):
        raise ValueError(f'Unexpected geometry: {before.shape}, {legacy.shape}')
    known = np.diff(legacy @ LUMA, axis=1)[4:-4, 4:-4].ravel()
    grad = np.diff(before @ LUMA, axis=1)
    scores = {(y, x): float(np.corrcoef(known, grad[4+y:-4+y, 4+x:-4+x].ravel())[0, 1])
              for y in range(-2, 3) for x in range(-2, 3)}
    center = scores.pop((0, 0))
    return {'unshifted': center, 'best_shifted': max(scores.values()),
            'passed': center >= .97 and center > max(scores.values())}


def evaluate(off, on, legacy, regions, fixed, out):
    registration = alignment(off, legacy)
    if on.shape != off.shape or not registration['passed']:
        raise ValueError(f'Invalid alignment: {registration}')
    zones = {}
    for name in SKIN + PROTECTED:
        x0, y0, x1, y1 = regions[name]
        area = np.zeros(off.shape[:2], bool)
        area[y0:y1, x0:x1] = True
        zones[name] = area
    skin = zones['cheek'] | zones['chest']
    target, candidate = off.copy(), off.copy()
    target[skin] = finish_spatial(off, on, STRENGTH)[skin]
    candidate[skin], clipped = render(chroma(off[skin]) + fixed, off[skin] @ LUMA)
    np.testing.assert_array_equal(target[~skin], candidate[~skin])
    base = chroma(off)
    desired, actual = chroma(target) - base, chroma(candidate) - base
    results = {key: stats(actual, desired, area) for key, area in zones.items()}
    for name in SKIN:
        result = results[name]
        result['editor_median'] = float(np.median(np.abs(desired[zones[name]]).mean(axis=1)))
        result['eligible'] = result['pixels'] >= 128 and result['editor_median'] >= 2
        result['passed'] = (result['eligible'] and .75 <= result['projection'] <= 1.25
                            and result['response_mae'] <= 3)
    for name in PROTECTED:
        results[name]['passed'] = results[name]['response_p95'] <= 1.5
    images = {name: save(out / f'{name}.png', rgb) for name, rgb in
              [('baseline', off), ('editor-local', target), ('fixed-replay', candidate)]}
    return {'alignment': registration, 'regions': results, 'clipped_fraction': clipped, 'images': images,
            'passed': all(row['passed'] for row in results.values())}


def run(base, out, default_anchor=None):
    if any(not path.is_absolute() or not path.as_posix().isascii() for path in (base, out)):
        raise ValueError('Use absolute ASCII paths')
    paths = {key: base / name for key, (name, _) in INPUTS.items()}
    for key, path in paths.items():
        if sha256(path.read_bytes()).hexdigest() != INPUTS[key][1]:
            raise ValueError(f'Changed input: {key}')
    fits = checked_json(base / 'playraw-mixed-light-operator-v1/results.json', FIT_SHA)['cases']
    fixed = np.mean([fits[fold]['constant'] for fold in ('0', '1')], axis=0)
    region_file = Path(__file__).with_name('darktable-cuckoo-legacy-regions.json')
    regions = checked_json(region_file, REGIONS_SHA)
    with Image.open(paths['legacy']) as image:
        legacy = np.array(image.convert('RGB'))
    source = paths['author'].read_bytes()
    if default_anchor is None:
        settings = states(paths['target'].read_bytes(), source)
    else:
        legacy, _ = read(default_anchor, False)
        if sha256(legacy.tobytes()).hexdigest() != DEFAULT_PIXELS:
            raise ValueError('Changed frozen default rendering')
        settings = default_states(source)
    out.mkdir()
    trials, renders = [], []
    # The vision helpers configure OpenMP for their own process. darktable uses its CLI defaults.
    editor_env = {key: value for key, value in os.environ.items() if key != 'OMP_NUM_THREADS'}
    for repeat in range(2):
        folder = out / str(repeat)
        folder.mkdir()
        pair = []
        for label, setting in zip(('off', 'on'), settings, strict=True):
            sidecar = folder / f'{label}.xmp'
            sidecar.write_bytes(setting)
            rgb, _ = read(export(paths['cli'], paths['raw'], sidecar, folder, label, False, env=editor_env), False)
            if label == 'off' and default_anchor is not None:
                if rgb.shape != legacy.shape or np.max(np.abs(rgb.astype(np.int16) - legacy)) > 1:
                    raise ValueError('Disabled module does not preserve the frozen default rendering')
            pair.append(rgb)
        renders.append(pair)
        trials.append(evaluate(*pair, legacy, regions, fixed, folder))
    max_difference = [int(np.max(np.abs(a.astype(np.int16) - b))) for a, b in zip(*renders, strict=True)]
    repeatable = max(max_difference) <= 1 and trials[0]['passed'] == trials[1]['passed']
    json_to(out / 'results.json', {'trials': trials, 'max_repeat_delta': max_difference,
                                   'baseline': 'default' if default_anchor is not None else 'original-history',
                                   'fit_sha256': FIT_SHA, 'fixed_chroma': fixed.tolist(),
                                   'settings_sha256': [sha256(s).hexdigest() for s in settings],
                                   'repeatable': repeatable, 'passed': repeatable and trials[0]['passed']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--default-anchor', type=Path)
    run(**vars(parser.parse_args()))
