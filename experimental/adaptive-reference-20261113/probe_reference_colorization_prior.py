"""Check whether a grayscale-predicted b* prior helps identify selective cooling."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from independent_coco import digest
from probe_reference_chroma_spread import BASE, ROOT
from probe_reference_cold_boundary import evaluate
from probe_reference_cold_ramp import BOUNDARY_SHA
from probe_reference_vision_cold import TOGGLE_SHA, TRANSFER_SHA


REVISION = '4f6009ed1495b1300231ebeb41cc4015557ddef7'
WEIGHTS = 'colorization_release_v2-9b330a0b.pth'
ASSETS = {
    'base_color.py': '81cea88bb484866478138440f08cc8d6c3e3731652fe57592900e418b278da1d',
    'eccv16.py': '524179cbf9676809a17eead0a5cf72bbc5de962db6bbbd1064407cfd315ae2a4',
    WEIGHTS: '9b330a0bae53f4ded77b1e23defbf78beaa09c10ebc4c4999e8e4f4a160b93f9',
    'LICENSE': '0df922ed99775db79b6f3c40b9d1d0b4d95194dcead6a5cac6cff2cf4006fade',
}
RGB_TO_XYZ = np.array([[.412453, .357580, .180423],
                       [.212671, .715160, .072169],
                       [.019334, .119193, .950227]])
D65 = np.array([.95047, 1., 1.08883])


def lightness_and_b(rgb):
    rgb = rgb.astype(np.float64) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = (linear @ RGB_TO_XYZ.T) / D65
    mapped = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * mapped[..., 1] - 16, 200 * (mapped[..., 1] - mapped[..., 2])


def load_eccv(assets):
    if (assets / WEIGHTS).stat().st_size != 128976165:
        raise ValueError('Official model size differs')
    for name, sha in ASSETS.items():
        if digest(assets / name) != sha:
            raise ValueError(f'Official asset changed: {name}')
    base, module = (assets / name for name in ('base_color.py', 'eccv16.py'))
    scoped = {'__name__': 'eccv16_reference_source'}
    exec(compile(base.read_text(encoding='utf-8'), str(base), 'exec'), scoped)
    source = module.read_text(encoding='utf-8').replace('from IPython import embed\n', '', 1)
    source = source.replace('from .base_color import *\n', '', 1)
    exec(compile(source, str(module), 'exec'), scoped)
    model = scoped['ECCVGenerator']()
    model.load_state_dict(torch.load(assets / WEIGHTS, weights_only=True, map_location='cpu'), strict=True)
    return model.to('cuda').eval().requires_grad_(False)


def matched_color(level, probability, observed, predicted):
    result = np.zeros(2, dtype=np.float64)
    support = 0
    for low in range(0, 100, 10):
        zone = (level >= low) & (level < low + 10)
        masks = zone & (probability >= .8), zone & (probability <= .2)
        count = min(int(mask.sum()) for mask in masks)
        if count < 64:
            continue
        support += count
        person, background = masks
        for index, plane in enumerate((observed - predicted, observed)):
            result[index] += count * (np.median(plane[person]) - np.median(plane[background]))
    return (result / support).tolist() if support >= 256 else [None, None], support


def fixed_results(first, second, photos, name):
    baseline = evaluate(first, name, photos)
    complete = [first[pair] for pair in photos if all(row[name] is not None for row in first[pair].values())]
    cutoff = None
    if len(complete) >= 2:
        pos = np.mean([scene['local'][name] for scene in complete])
        neg = np.mean([scene[role][name] for scene in complete for role in ('base', 'whole')])
        cutoff = float((pos + neg) / 2)
    counts, false, cases = dict(hard=0, wide=0), 0, {}
    for photo, versions in second.items():
        rows = {}
        for role, row in versions.items():
            value = row[name]
            chosen = None if value is None or cutoff is None else value > cutoff
            if chosen is True:
                if row['expected_local']:
                    counts[role] += 1
                else:
                    false += 1
            rows[role] = chosen
        cases[photo] = rows
    return dict(first=baseline, threshold=cutoff, second_hits=counts, second_false_triggers=false,
                second_predictions=cases, passed=baseline['passed'] and min(counts.values()) >= 3 and false == 0)


def load_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Frozen evidence changed: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def measure(model, out, cohort, image_path, probability_path, rgb_hash, expected):
    features = {}
    for group, versions in cohort.items():
        rows = {}
        for role, entry in versions.items():
            path, mask = image_path(group, role), probability_path(group, role)
            identity = rgb_hash(entry)
            if digest(path) != identity or digest(mask) != entry['probability_sha256']:
                raise ValueError('Reference or automatic probability changed')
            with Image.open(path) as image:
                if image.size != (256, 256) or image.mode != 'RGB':
                    raise ValueError('Expected 256x256 sRGB')
                level, actual = lightness_and_b(np.array(image))
            tensor = torch.from_numpy(level.astype(np.float32))[None, None].to('cuda')
            with torch.inference_mode():
                predicted = model(tensor)[0, 1].cpu().numpy()
            if predicted.shape != level.shape or not np.isfinite(predicted).all():
                raise ValueError('Unexpected color prior')
            prior_path = out / f'{group}-{role}-prior-b.npy'
            np.save(prior_path, predicted, allow_pickle=False)
            person = np.load(mask, allow_pickle=False)
            if person.shape != level.shape or not np.isfinite(person).all():
                raise ValueError('Unexpected person probability')
            measured, support = matched_color(level, person, actual, predicted)
            rows[role] = dict(prior=measured[0], raw=measured[1], support=support, rgb_sha256=identity,
                              probability_sha256=entry['probability_sha256'], prior_sha256=digest(prior_path),
                              expected_local=expected(group, entry, role))
        features[group] = rows
    return features


def run(out):
    started = time.monotonic()
    docs = ROOT / 'docs/research'
    toggle = load_json(docs / '2026-11-18-local-edit-toggle-evidence.json', TOGGLE_SHA)
    boundary = load_json(docs / '2026-11-18-reference-cold-boundary-evidence.json', BOUNDARY_SHA)
    transfer = load_json(docs / '2026-11-18-reference-cold-ramp-transfer-evidence.json', TRANSFER_SHA)
    photos = [pair for pair, row in toggle['cases'].items() if row['gates']['positive']]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_eccv(ROOT / '.tools/ECCV16-colorization-audit')
    out.mkdir(exist_ok=False)
    first = measure(model, out, boundary['groups'],
                    lambda group, role: BASE / 'local-edit-toggle-v1' / group / f'{role}-reference.png',
                    lambda group, role: BASE / 'reference-cold-boundary-v1' / f'{group}-{role}.npy',
                    lambda entry: entry['reference_sha256'], lambda group, entry, role: group in photos and role == 'local')
    transfer_dir = BASE / 'reference-cold-ramp-transfer-v1'
    second = measure(model, out, {key: row['versions'] for key, row in transfer['cases'].items()},
                     lambda group, role: transfer_dir / f'{group}-{role}.png',
                     lambda group, role: transfer_dir / f'{group}-{role}.npy',
                     lambda entry: entry['rgb_sha256'], lambda group, entry, role: entry['expected_local'])
    comparisons = {name: fixed_results(first, second, photos, name) for name in ('prior', 'raw')}
    elapsed = time.monotonic() - started
    if elapsed >= 90:
        raise TimeoutError('Fixed feature screen exceeded its budget')
    source = Path(__file__)
    result = dict(first=first, second=second, comparisons=comparisons, seconds=elapsed, assets=ASSETS,
                  author_revision=REVISION, sources=dict(toggle=TOGGLE_SHA, boundary=BOUNDARY_SHA,
                                                         transfer=TRANSFER_SHA),
                  script_sha256=digest(source), protocol_sha256=digest(source.with_name(
                      'reference-colorization-prior-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: {'first': {key: v for key, v in row['first'].items() if key != 'results'},
                             'second_hits': row['second_hits'], 'second_false': row['second_false_triggers'],
                             'passed': row['passed']} for name, row in comparisons.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
