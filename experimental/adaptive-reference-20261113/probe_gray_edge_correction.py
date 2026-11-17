"""Test a gradient-based color correction using only the two normal input photos."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import region_delta, target_regions
from independent_coco import digest
from probe_multilum_chromaticity import linear_rgb
from probe_psbench_action_rgb_transfer import score
from probe_psbench_teal_portrait import verify_source
from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, decision, projection, read
from reference_color_app.media import decode_image


PARENT_SHA = 'd7dac2996d03cdf8d5a08c3e0200272772bfd8c6f7fe0d81931c84db01cd33dc'
LUMA = np.array([.2126, .7152, .0722])
SMOOTH = np.array([1, 4, 6, 4, 1]) / 16


def linear_edge_chroma(linear):
    smooth_x = sum(weight * linear[:, i:linear.shape[1] - 4 + i] for i, weight in enumerate(SMOOTH))
    smooth = sum(weight * smooth_x[i:linear.shape[0] - 4 + i] for i, weight in enumerate(SMOOTH))
    dx = smooth[1:-1, 2:] - smooth[1:-1, :-2]
    dy = smooth[2:, 1:-1] - smooth[:-2, 1:-1]
    energy = np.mean((dx * dx + dy * dy) ** 3, axis=(0, 1)) ** (1 / 6)
    if np.min(energy) < 1e-8:
        return None
    logs = np.log(energy)
    return logs - logs.mean()


def edge_chroma(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    return linear_edge_chroma(linear_rgb(np.asarray(image))) if min(image.size) >= 7 else None


def correction(target, reference, baseline):
    zero = dict(gain=[0., 0., 0.], clipped_channel_fraction=0.)
    if np.array_equal(target, reference):
        return target.copy(), zero
    source, look, rendered = map(edge_chroma, (target, reference, baseline))
    if any(value is None for value in (source, look, rendered)):
        return baseline.copy(), zero
    gain = np.clip(.8 * (look - source) - (rendered - source), -.15, .15)
    raw = linear_rgb(baseline)
    adjusted = raw * np.exp(gain)
    luma = adjusted @ LUMA
    luminance_keep = np.divide(raw @ LUMA, luma, out=np.ones_like(luma), where=luma > 0)
    adjusted *= luminance_keep[..., None]
    clipped = float(np.mean(adjusted > 1))
    adjusted = adjusted.clip(0, 1)
    encoded = np.where(adjusted <= .0031308, 12.92 * adjusted, 1.055 * adjusted ** (1 / 2.4) - .055)
    return np.rint(encoded * 255).astype(np.uint8), dict(gain=gain.tolist(), clipped_channel_fraction=clipped)


def save_rgb(path, rgb):
    Image.fromarray(rgb).save(path)
    return digest(path)


def controlled(out, parent):
    cases = []
    for case in parent['cases']:
        target = read(case['input_file'])
        pair = case['pair']
        folder = out / pair
        folder.mkdir()
        inputs = {row['name']: row for row in case['variants']}
        prediction = {}
        diagnostics = {}
        for name in ('identity', *BACKGROUND, *GLOBAL):
            variant = inputs[name]
            ref = read(variant['reference'])
            baseline = read(variant['methods']['adaptive']['file'])
            rgb, debug = correction(target, ref, baseline)
            prediction[name] = dict(baseline=baseline, candidate=rgb)
            diagnostics[name] = dict(**debug, sha256=save_rgb(folder / f'{name}.png', rgb))

        mask_name = pair.split('-')[0] + '-human-mask.png'
        mask = BASE / 'independent-coco' / mask_name
        if digest(mask) != case['parents'][mask_name]:
            raise ValueError(f'{pair}: scoring mask changed')
        with Image.open(mask) as image:
            human = np.asarray(image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        regions = target_regions(target, human)
        anchor = prediction['identity']
        rows = {}
        for name, values in prediction.items():
            deltas = {key: value.astype(np.int16) - anchor[key] for key, value in values.items()}
            rows[name] = dict(regions={key: {r: region_delta(value, anchor[key], m) for r, m in regions.items()}
                                       for key, value in values.items()},
                              projection={key: projection(deltas['candidate'], deltas['baseline'], m)
                                          for key, m in regions.items()}, **diagnostics[name])
        cases.append(dict(pair=pair, variants=rows, original_reference_drift=region_delta(
            anchor['candidate'], anchor['baseline'], regions['whole'])['rgb_mae']))
    return cases


def authored(out, folder, evidence):
    baseline = json.loads(evidence.read_text(encoding='utf-8'))
    if baseline['files'] != verify_source(folder, '105') or baseline['strength'] != .8:
        raise ValueError('Unexpected PSBench source or strength')
    source, edited = [decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end')]
    crops = {name: slice(*pair) for name, pair in baseline['crop_columns'].items()}
    results = {}
    for target, reference in [('portrait', 'tower'), ('tower', 'portrait')]:
        path = folder / 'action-pair-v1' / f'{target}.png'
        if digest(path) != baseline['predictions'][target]:
            raise ValueError(f'{target}: PSBench baseline changed')
        original, grade, current = source[:, crops[target]], edited[:, crops[target]], decode_image(path)
        refined, debug = correction(original, edited[:, crops[reference]], current)
        sha = save_rgb(out / f'psbench-{target}.png', refined)
        expected = original + np.rint((grade.astype(np.int16) - original) * .8).astype(np.int16)
        metrics = score(original, expected, dict(original=original, current13=current, candidate=refined),
                        np.ones(original.shape[:2], dtype=bool))
        gate = all(row['methods']['candidate']['mae'] <= min(
            row['methods']['original']['mae'], row['methods']['current13']['mae'] - 1
        ) for row in metrics.values()) and metrics['whole']['methods']['candidate']['projection'] > 0
        results[target] = dict(metrics=metrics, gate=gate, sha256=sha, **debug)
    return results


def run(out):
    manifest = PARENT / 'results.json'
    if digest(manifest) != PARENT_SHA:
        raise ValueError('Background intervention baseline changed')
    parent = json.loads(manifest.read_text(encoding='utf-8'))
    if not parent['finished'] or len(parent['cases']) != 6 or parent['strength'] != .8:
        raise ValueError('Unexpected background intervention dataset')
    out.mkdir(exist_ok=False)
    cases = controlled(out, parent)
    evidence = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-psbench-action-pair-evidence.json'
    comparisons = authored(out, BASE / 'psbench-shared-local-v1', evidence)
    result = dict(controlled=decision(cases), cases=cases, authored=comparisons, parent_sha256=PARENT_SHA,
                  model_sha256=parent['model_sha256'], pair_sha256=digest(evidence),
                  script_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'gray-edge-correction-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(controlled=result['controlled'], authored=comparisons)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
