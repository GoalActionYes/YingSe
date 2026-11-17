"""Bound uniform attenuation of the frozen RGB8 background controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, digest, read, target_regions

PARENT_SHA256 = 'd7dac2996d03cdf8d5a08c3e0200272772bfd8c6f7fe0d81931c84db01cd33dc'


def projection_lower(delta):
    delta = delta.astype(np.float64)
    squared = np.square(delta).sum()
    assert squared > 0
    return max(0., .90 - float(np.abs(delta).sum() / squared))


def rounded_controls(images, target, masks, changes):
    shifts = {name: (image - target).astype(np.int32) for name, image in images.items()}
    target_scaled = target.astype(np.int32) * 100
    probes = [(name, changes[name].astype(np.float64)) for name in GLOBAL[2:]]
    person = masks['person']
    neutral = masks['neutral_proxy']
    eligible = []
    for step in range(101):
        rounded = {name: np.rint((target_scaled + step * shift) / 100).astype(np.int16)
                   for name, shift in shifts.items()}
        delta = {name: image - rounded['identity'] for name, image in rounded.items()}
        if not all(.90 <= np.sum(delta[name] * base) / np.square(base).sum() <= 1.10
                   for name, base in probes):
            continue
        passed = 0
        for background, overall in zip(BACKGROUND, GLOBAL[:2], strict=True):
            if person.any():
                original = np.abs(changes[background][person]).mean()
                changed = np.abs(delta[background][person]).mean()
                base = changes[overall][person].astype(np.float64)
                retained = np.sum(delta[overall][person] * base) / np.square(base).sum()
                passed += int(changed <= min(.8 * original, original - .25) and retained >= .90)
        eligible.append((step, passed, sum(float(np.abs(delta[name][neutral]).mean()) for name in BACKGROUND)))
    assert eligible and eligible[-1][0] == 100
    best_neutral = min(eligible, key=lambda item: item[2])
    return dict(eligible_steps=[item[0] for item in eligible], person_max=max(item[1] for item in eligible),
                neutral_min_step=best_neutral[0], neutral_min_sum=best_neutral[2])


def run(out):
    parent_file = PARENT / 'results.json'
    assert digest(parent_file) == PARENT_SHA256
    parent = json.loads(parent_file.read_text(encoding='utf-8'))
    assert parent['strength'] == .8 and len(parent['cases']) == 6
    cases = []
    for case in parent['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        mask_name = pair.split('-')[0] + '-human-mask.png'
        path = BASE / 'independent-coco' / mask_name
        assert digest(path) == case['parents'][mask_name]
        with Image.open(path) as image:
            person = np.asarray(image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        masks = target_regions(target, person)
        variants = {row['name']: row for row in case['variants']}
        images = {name: read(variants[name]['methods']['adaptive']['file']).astype(np.int16)
                  for name in ['identity', *BACKGROUND, *GLOBAL]}
        changes = {name: image - images['identity'] for name, image in images.items()}
        whole_lower = {name: projection_lower(changes[name]) for name in GLOBAL[2:]}
        common_lower = max(whole_lower.values())
        row = dict(pair=pair, whole_lower=whole_lower, directions=[],
                   rounded=rounded_controls(images, target, masks, changes))
        for background, overall in zip(BACKGROUND, GLOBAL[:2], strict=True):
            delta = changes[background]
            neutral = np.abs(delta[masks['neutral_proxy']])
            result = dict(name=background, neutral_mae=float(neutral.mean()),
                          neutral_scaled_mae_lower=float(np.maximum(common_lower * neutral - 1, 0).mean()))
            if masks['person'].any():
                interference = float(np.abs(delta[masks['person']]).mean())
                assert interference > 0
                upper = min(1., (min(.8 * interference, interference - .25) + 1) / interference)
                lower = max(common_lower, projection_lower(changes[overall][masks['person']]))
                result.update(person_mae=interference, lower=lower, upper=upper, possible=lower <= upper)
            row['directions'].append(result)
        cases.append(row)
    directions = [d for case in cases for d in case['directions']]
    possible = [d['possible'] for d in directions if 'possible' in d]
    assert len(possible) == 8
    neutral_total = sum(d['neutral_mae'] for d in directions)
    neutral_gain_upper = 1 - sum(d['neutral_scaled_mae_lower'] for d in directions) / neutral_total
    rounded_neutral_gain = 1 - sum(c['rounded']['neutral_min_sum'] for c in cases) / neutral_total
    report = dict(parent_sha256=PARENT_SHA256, source_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('uniform-strength-bound-protocol.md')),
                  verified_baseline_outputs=42, person_possible=sum(possible),
                  person_total=len(possible), neutral_gain_upper=neutral_gain_upper,
                  rounded_person_max=sum(c['rounded']['person_max'] for c in cases),
                  rounded_neutral_gain=rounded_neutral_gain, cases=cases)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
