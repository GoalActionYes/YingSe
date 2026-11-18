"""Run the frozen portable skin protector on frozen 13 image controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageFilter

from audit_bg_real_skin_regions import BASE, LUMA, PROTOCOL, ROOT, digest, numbers, read

SKIN_SHA = '05e190ebb42a4047fd6bdd532ea5946a3a6394a98ac5d30e8d7065358f252b83'
TOGGLE_SHA = '437681478dfde0362c953b0210598fbc4f23f1d043e7be8514f4a49d4df5857a'
ROLES = ('base', 'local', 'whole')


def protector(path):
    assert digest(path) == SKIN_SHA
    spec = importlib.util.spec_from_file_location('frozen_yingse_skin', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    Image.fromarray(value).save(path)
    return digest(path)


def projection(actual, expected):
    return float(np.sum(actual * expected) / np.sum(expected ** 2))


def chroma(delta):
    return delta - (delta @ LUMA)[..., None]


def run(module_path, out):
    start = time.monotonic()
    module = protector(module_path)
    paths = ROOT / 'docs/research'
    evidence = paths / '2026-11-18-bg-real-pixel-pilot-evidence.json'
    inputs = json.loads(evidence.read_bytes())
    toggle_path = paths / '2026-11-18-local-edit-toggle-evidence.json'
    assert digest(toggle_path) == TOGGLE_SHA
    toggle = json.loads(toggle_path.read_bytes())
    parent_dir = BASE / 'background-intervention-20261116-v2'
    parent_file = parent_dir / 'results.json'
    assert digest(parent_file) == inputs['parent_sha256'] == toggle['parent_sha256']
    parent = {row['pair']: row for row in json.loads(parent_file.read_bytes())['cases']}
    assert inputs['checkpoint_sha256'] == toggle['model_sha256']
    same = np.full((256, 256, 3), 128, np.uint8)
    np.testing.assert_array_equal(module.protect_image(same, same.copy())[0], same)
    out.mkdir()
    photo = {}
    local = {}
    hashes = {}
    for key, row in inputs['cases'].items():
        pair = row['target_pair']
        entry = parent[pair]['input_file']
        original = read(parent_dir/entry['path'], entry['sha256']).astype(np.uint8)
        weights, _, debug = module._predict(original)
        grades = {role: read(BASE/'bg-real-pixel-pilot'/f'{key}-{role}.png', row['outputs'][role])
                  for role in ('C1', 'F1')}
        protected = {}
        for role, grade in grades.items():
            image, _ = module.protect_image(original, grade.astype(np.uint8))
            name = f'{key}-{role}.png'
            hashes[name] = save(out/name, image)
            protected[role] = image.astype(np.int16)
        photo[pair] = (grades, protected, weights, debug)
    for pair, row in toggle['cases'].items():
        folder = BASE/'local-edit-toggle-v1'/pair
        files = row['files']
        original = read(folder/'target.png', files['target.png']).astype(np.uint8)
        grades, protected = {}, {}
        for role in ROLES:
            name = f'{role}-current13.png'
            grades[role] = read(folder/name, files[name])
            image, _ = module.protect_image(original, grades[role].astype(np.uint8))
            name = f'{pair}-{role}.png'
            hashes[name] = save(out/name, image)
            protected[role] = image.astype(np.int16)
        local[pair] = (grades, protected)

    patches = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                         PROTOCOL.read_text(encoding='utf-8'), re.MULTILINE)
    assert len(patches) == 9
    skin, metrics, skin_areas = {}, {}, {}
    for target, reference, label, *bounds in patches:
        pair = f'{target}-{reference}'
        left, top, right, bottom = map(int, bounds)
        area = np.s_[top:bottom, left:right]
        grade, protected, weight, debug = photo[pair]
        first, second = (numbers((renders['F1']-renders['C1'])[area]) for renders in (grade, protected))
        direct = label not in ('雪地', '浅墙', '白球服', '投影幕中的脸')
        row = dict(baseline=first, protected=second, weight_mean=float(weight[area].mean()),
                   detected=debug['skin_detected'], direct=direct)
        metrics.setdefault(pair, {})[label] = row
        if direct:
            skin.setdefault(pair, []).append(row)
            skin_areas.setdefault(pair, []).append(area)

    style = {}
    for pair, (grade, protected) in local.items():
        folder = BASE/'local-edit-toggle-v1'/pair
        files = toggle['cases'][pair]['files']
        expected = {role: read(folder/f'{role}-expected.png', files[f'{role}-expected.png']) for role in ROLES}
        target_id = pair.split('-')[0]
        mask_name = f'{target_id}-human-mask.png'
        mask_path = BASE/'independent-coco'/mask_name
        assert digest(mask_path) == parent[pair]['parents'][mask_name]
        with Image.open(mask_path) as label:
            mask = label.convert('L').resize((256, 256), Image.Resampling.NEAREST).point(lambda v: 255 if v else 0)
            core = np.array(mask.filter(ImageFilter.MaxFilter(25))) == 0
            person = mask.getbbox() is not None
        actual = [{role: images[role].astype(float)-images['base'] for role in ('local', 'whole')}
                  for images in (grade, protected)]
        truth = {role: expected[role].astype(float)-expected['base'] for role in ('local', 'whole')}
        row = dict(whole=[projection(a['whole'], truth['whole']) for a in actual])
        if person:
            row['background'] = [projection(a['local'][core], truth['local'][core]) for a in actual]
        else:
            row['no_person'] = bool(np.array_equal(protected['base'], protected['local'])
                                    and np.array_equal(protected['base'], grade['base']))
        if pair in skin_areas:
            def flatten(image):
                return np.concatenate([image[area].reshape(-1, 3) for area in skin_areas[pair]])
            direction = chroma(flatten(truth['whole']))
            row['skin_whole'] = [projection(chroma(flatten(a['whole'])), direction) for a in actual]
        style[pair] = row
    assert len(skin) == 3 and sum(map(len, skin.values())) == 5 and len(hashes) == 26
    before_after = [(row['baseline']['chroma_mae'], row['protected']['chroma_mae'])
                    for rows in skin.values() for row in rows]
    improved = sum(b <= .85*a for a, b in before_after)
    gates = dict(spill=improved >= 4 and all(b <= a+.25 for a, b in before_after),
                 skin_style=all(b > 0 and b >= .90*a for a, b in (style[p]['skin_whole'] for p in skin)),
                 background=all(b >= a-.10 for a, b in (row['background'] for row in style.values()
                                                        if 'background' in row)),
                 whole=all(not .75 <= a <= 1.25 or .75 <= b <= 1.25
                           for a, b in (row['whole'] for row in style.values())),
                 no_person=all(row.get('no_person', True) for row in style.values()),
                 same_image=True)
    assert time.monotonic()-start <= 90
    result = dict(metrics=metrics, style=style, improved=improved, gates=gates, proceed=all(gates.values()),
                  output_sha256=hashes, source_sha256=digest(Path(__file__)), skin_source_sha256=SKIN_SHA,
                  model_sha256=module.MODEL_SHA256, input_evidence_sha256=digest(evidence),
                  skin_regions_protocol_sha256=digest(PROTOCOL), toggle_sha256=TOGGLE_SHA)
    (out/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(improved, gates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('skin_source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.skin_source.resolve(), args.out)
