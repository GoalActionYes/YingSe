"""Run the upstream VCG preprocessing on frozen two-image portraits and audit its checkpoint header."""

from argparse import ArgumentParser
import ast
from collections import Counter
from math import prod
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, save


UTIL_SHA = 'ae5a3222dbba15acc4a890b177d54efd1802647704a9be13dd8ff260c0bc7904'
HEADER_SHA = 'ae5f5dba740c72909745d34ae464d5f69d50f8b46b1092a3d93b43067bd6d5b8'
FILE_SIZE = 4_119_430_034
CASES = ('01-from-02', '02-from-01')
STATES = ('off', 'on')


def upstream_preprocess(path):
    if digest(path) != UTIL_SHA:
        raise ValueError('Unexpected author source')
    source = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name in ('vars', 'transfer', 'preprocess')]
    if [node.name for node in selected] != ['vars', 'transfer', 'preprocess']:
        raise ValueError('Unexpected preprocessing source')
    namespace = {'np': np, 'Image': Image}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['preprocess']


def checkpoint_index(path):
    if digest(path) != HEADER_SHA:
        raise ValueError('Unexpected bounded checkpoint range')
    raw = path.read_bytes()
    length = int.from_bytes(raw[:8], 'little')
    if length < 2 or length + 8 > len(raw):
        raise ValueError('Missing full safetensors header')
    metadata = json.loads(raw[8:8 + length])
    groups, blocks = {}, []
    for key, item in metadata.items():
        if key == '__metadata__':
            continue
        start, end = item['data_offsets']
        dtype = item['dtype']
        size = {'F16': 2, 'F32': 4, 'BF16': 2, 'I64': 8}[dtype]
        if end < start or end - start != size * prod(item['shape']):
            raise ValueError(f'Inconsistent tensor storage for {key}')
        group = groups.setdefault(key.split('.')[0], {'count': 0, 'bytes': 0, 'dtypes': Counter()})
        group['count'] += 1
        group['bytes'] += end - start
        group['dtypes'][dtype] += 1
        blocks.append((start, end))
    if set(groups) != {'clip_vision', 'vae', 'referencenet', 'l_diffuser'}:
        raise ValueError('Unexpected component names')
    position = 0
    for start, end in sorted(blocks):
        if start != position:
            raise ValueError('Gap or overlap in checkpoint layout')
        position = end
    if position + length + 8 != FILE_SIZE:
        raise ValueError('Indexed bytes do not match published file length')
    return {'header_bytes': length, 'file_bytes': FILE_SIZE, 'groups': groups}


def run(source, out):
    official = upstream_preprocess(source / 'util.py')
    checkpoint = checkpoint_index(source / 'header.bin')
    prior = BASE / 'gros-film-b-cross-v2'
    manifest = checked_json(prior / 'results.json', BASELINE_SHA)
    ledger = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    records = {}
    out.mkdir()
    for direction in CASES:
        person, donor = direction.split('-from-')
        target = checked_rgb(prior / f'{person}-cold-off.png', manifest['inputs'][f'{person}-cold']['hashes']['off'])
        expected = checked_rgb(prior / f'{direction}-expected.png', manifest['cases'][direction]['expected_sha'])
        areas = masks(ledger[person]['cold'], ledger['screen'], target.shape[:2])
        scores, hashes = {}, {}
        for arm, skip in [('default', False), ('ncc', True)]:
            outputs = {}
            for light in LIGHTS:
                outputs[light] = {}
                for state in STATES:
                    name = f'{donor}-{light}'
                    reference = checked_rgb(prior / f'{name}-{state}.png', manifest['inputs'][name]['hashes'][state])
                    resized = np.asarray(Image.fromarray(reference).resize((512, 512)))
                    native, _ = official([target], resized, 512, skip)
                    if skip:
                        np.testing.assert_array_equal(native[0], target)
                    grade = finish_spatial(target, native[0], STRENGTH)
                    outputs[light][state] = grade
                    filename = f'{direction}-{arm}-{light}-{state}.png'
                    hashes[filename] = save(out / filename, grade)
            scores[arm] = score(target, expected, outputs, areas)
        records[direction] = {'scores': scores, 'hashes': hashes}
    json_to(out / 'results.json', {'source_sha256': UTIL_SHA, 'range_sha256': HEADER_SHA,
                                   'checkpoint_index': checkpoint, 'cases': records})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
