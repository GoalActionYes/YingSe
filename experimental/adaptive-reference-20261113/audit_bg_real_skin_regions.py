"""Measure fixed, source-annotated patches on two sets of existing grades."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


PROTOCOL = Path(__file__).with_name('bg-real-skin-regions-protocol.md')
ROOT = PROTOCOL.resolve().parents[2]
BASE = ROOT / 'outputs/adaptive-reference-20261113'
LUMA = np.array([.299, .587, .114])


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f'Unexpected file identity: {path.name}')
    with Image.open(path) as image:
        if image.size != (256, 256) or image.mode != 'RGB':
            raise ValueError(f'Unexpected image domain: {path.name}')
        return np.array(image, dtype=np.int16)


def numbers(delta):
    luma = delta @ LUMA
    return dict(pixels=int(luma.size), rgb_mae=float(np.abs(delta).mean()),
                chroma_mae=float(np.abs(delta-luma[..., None]).mean()),
                luma_mae=float(np.abs(luma).mean()), rgb_mean=delta.mean(axis=(0, 1)).tolist())


def run(out):
    docs = ROOT / 'docs/research'
    prior = docs / '2026-11-18-bg-real-pixel-pilot-evidence.json'
    candidate = docs / '2026-11-18-bg-real-color-directions-evidence.json'
    prior_doc, candidate_doc = [json.loads(path.read_bytes()) for path in (prior, candidate)]
    assert not candidate_doc['proceed'] and candidate_doc['held_sha256'] == digest(prior)
    parent_path = BASE / 'background-intervention-20261116-v2/results.json'
    assert digest(parent_path) == prior_doc['parent_sha256']
    parent = {row['pair']: row for row in json.loads(parent_path.read_bytes())['cases']}
    patches = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                         PROTOCOL.read_text(encoding='utf-8'), re.MULTILINE)
    assert len(patches) == 9
    out.mkdir()
    data = {}
    direct = {}
    for key, row in prior_doc['cases'].items():
        pair = row['target_pair']
        folder = BASE / 'background-intervention-20261116-v2'
        source = parent[pair]['input_file']
        target = read(folder / source['path'], source['sha256'])
        annotated = Image.fromarray(target.astype(np.uint8))
        draw = ImageDraw.Draw(annotated)
        variant = candidate_doc['photo'][key]
        assert variant['target_pair'] == pair
        first, second = [read(BASE/'bg-real-pixel-pilot'/f'{key}-{role}.png', row['outputs'][role])
                         for role in ('C1', 'F1')]
        filtered_first, filtered_second = [
            read(BASE/'bg-real-color-directions'/f'{key}-{role}.png', variant['output'][role]['sha256'])
            for role in ('C1', 'F1')]
        records = {}
        for index, (target_id, reference_id, label, *bounds) in enumerate(patches, start=1):
            if f'{target_id}-{reference_id}' != pair:
                continue
            left, top, right, bottom = map(int, bounds)
            assert 0 <= left < right <= 256 and 0 <= top < bottom <= 256
            crop = np.s_[top:bottom, left:right]
            before = numbers(second[crop] - first[crop])
            after = numbers(filtered_second[crop] - filtered_first[crop])
            records[label] = dict(index=index, baseline=before, projection=after)
            if label not in ('雪地', '浅墙', '白球服', '投影幕中的脸'):
                direct.setdefault(pair, []).append(before['chroma_mae'] > 1)
            draw.rectangle((left-1, top-1, right, bottom), outline=(255, 238, 0))
            draw.text((right+1, top), str(index), fill=(255, 238, 0), stroke_width=1, stroke_fill=(0, 0, 0))
        annotated.save(out / f'{pair}-annotated.png')
        data[pair] = records
    assert sum(map(len, data.values())) == 9 and len(direct) == 3
    count = sum(any(patches) for patches in direct.values())
    result = dict(records=data, direct_skin_groups_over_one=count, direct_skin_total=len(direct),
                  proceeds_as_development_measurement=count >= 2,
                  source_sha256=digest(Path(__file__)), protocol_sha256=digest(PROTOCOL),
                  baseline_evidence_sha256=digest(prior), candidate_evidence_sha256=digest(candidate))
    (out/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
