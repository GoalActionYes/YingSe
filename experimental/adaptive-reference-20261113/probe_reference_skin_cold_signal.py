"""Test a single-reference skin tint score on fixed color controls."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from audit_bg_real_skin_regions import BASE, ROOT, digest
from probe_bg_real_legacy_skin import ROLES, TOGGLE_SHA, protector


def feature(module, path, expected_sha):
    assert digest(path) == expected_sha
    with Image.open(path) as image:
        rgb = np.array(ImageOps.exif_transpose(image).convert('RGB'))
    weights, _, _ = module._predict(rgb)
    blue_minus_red = rgb[..., 2].astype(np.int16) - rgb[..., 0]
    # In the legacy predictor a large reference can have a reduced mask.
    if weights.shape != rgb.shape[:2]:
        weights = cv2.resize(weights, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    selected = weights >= .8
    count = int(selected.sum())
    return dict(skin_pixels=count,
                skin=float(np.median(blue_minus_red[selected])) if count >= 16 else None,
                full=float(np.median(blue_minus_red)))


def evaluate(people, empty, photos, field):
    def threshold(training):
        return max((entry[role][field] for entry in training for role in ('base', 'local')
                    if entry[role][field] is not None), default=None)

    def classify(versions, cutoff):
        return {name: cutoff is not None and row[field] is not None and row[field] > cutoff
                for name, row in versions.items()}

    folds = {}
    for pair, photos_in_group in people.items():
        cutoff = threshold(row for key, row in people.items() if key != pair)
        folds[pair] = dict(threshold=cutoff, decisions=classify(photos_in_group, cutoff))
    cutoff = threshold(people.values())
    empty_predictions = {pair: classify(photos, cutoff) for pair, photos in empty.items()}
    flips = {}
    for fold, entry in folds.items():
        flips[fold] = []
        for pair, rows in photos.items():
            predicted = classify(rows, entry['threshold'])
            if not predicted['C1'] and predicted['F1']:
                flips[fold].append(pair)
    hits = sum(row['decisions']['whole'] for row in folds.values())
    false_positives = sum(row['decisions'][role] for row in folds.values() for role in ('base', 'local'))
    empty_positives = sum(sum(row.values()) for row in empty_predictions.values())
    num_flips = sum(map(len, flips.values()))
    return dict(folds=folds, empty_threshold=cutoff, empty_decisions=empty_predictions, bg_flips=flips,
                hits=hits, false_positives=false_positives, empty_positives=empty_positives,
                bg_flips_count=num_flips, proceed=hits>=3 and not (false_positives or empty_positives or num_flips))


def run(skin, out):
    start = time.monotonic()
    module = protector(skin)
    docs = ROOT/'docs/research'
    toggle_path = docs/'2026-11-18-local-edit-toggle-evidence.json'
    bg_path = docs/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    assert digest(toggle_path) == TOGGLE_SHA
    toggle = json.loads(toggle_path.read_bytes())
    swap = json.loads(bg_path.read_bytes())
    people, empty = {}, {}
    for pair, row in toggle['cases'].items():
        folder = BASE/'local-edit-toggle-v1'/pair
        values = {}
        for role in ROLES:
            name = f'{role}-reference.png'
            values[role] = feature(module, folder/name, row['files'][name])
        (people if row['reference_person'] else empty)[pair] = values
    photos = {}
    for key, row in swap['cases'].items():
        photos[key] = {role: feature(module, BASE/'bg-real-pixel-inputs'/row['input'][role]['name'],
                                    row['input'][role]['sha256']) for role in ('C1', 'F1')}
    assert (len(people), len(empty), len(photos)) == (4, 2, 4)
    scores = {field: evaluate(people, empty, photos, field) for field in ('skin', 'full')}
    source = Path(__file__)
    result = dict(people=people, empty=empty, bg=photos, scores=scores, model_sha256=module.MODEL_SHA256,
                  toggle_sha256=TOGGLE_SHA, bg_sha256=digest(bg_path), source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('reference-skin-cold-signal-protocol.md')))
    assert time.monotonic()-start < 60
    with out.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print({field: [row['hits'], row['false_positives'], row['empty_positives'], row['bg_flips_count']]
           for field, row in scores.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('skin_source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.skin_source.resolve(), args.out)
