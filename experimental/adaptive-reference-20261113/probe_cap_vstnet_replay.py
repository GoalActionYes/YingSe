"""Cross replay reference features and automatic labels for fixed background controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from probe_cap_vstnet import BACKGROUND, BASE, PARENT, PARENT_SHA, ROOT, digest, finish_spatial, load_candidate, read
from background_intervention import target_regions

PRIOR = ROOT / 'docs/research/2026-11-18-cap-vstnet-screen-evidence.json'


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = PARENT / 'results.json'
    assert digest(parent) == PARENT_SHA
    baseline = json.loads(parent.read_text(encoding='utf-8'))
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    original_runs = BASE / 'cap-vstnet-screen-v2'
    encode, render, revision = load_candidate('cuda', replay=True)
    assert revision == prior['segmentation_revision']
    out.mkdir(exist_ok=False)
    cases = []
    start = time.monotonic()
    for case, prior_case in zip(baseline['cases'], prior['cases'], strict=True):
        pair = case['pair']
        assert pair == prior_case['pair']
        target = read(case['input_file'])
        mask = target_regions(target, np.zeros(target.shape[:2], dtype=bool))['neutral_proxy']
        target_features = encode(target)
        variants = {v['name']: v for v in case['variants']}
        reference_features, reference_labels = encode(read(variants['identity']['reference']))
        identity = finish_spatial(target, render(*target_features, reference_features, reference_labels), .8)
        folder = out / pair
        folder.mkdir()

        def saved(name, output):
            path = folder / f'{name}.png'
            Image.fromarray(output).save(path)
            return digest(path)

        def bridge(name, output):
            path = original_runs / pair / f'{name}-candidate.png'
            assert digest(path) == prior_case['variants'][name]['sha256']
            with Image.open(path) as previous:
                difference = float(np.abs(np.asarray(previous).astype(np.int16) - output).mean())
            assert difference <= .05, (pair, name, difference)
            return difference

        identity_hash = saved('identity', identity)
        identity_bridge = bridge('identity', identity)
        directions = {}
        for name in BACKGROUND:
            changed_features, changed_labels = encode(read(variants[name]['reference']))
            conditions = {
                'full': (changed_features, changed_labels),
                'features_only': (changed_features, reference_labels),
                'labels_only': (reference_features, changed_labels),
            }
            outputs = {k: finish_spatial(target, render(*target_features, *v), .8) for k, v in conditions.items()}
            hashes = {k: saved(f'{name}-{k}', v) for k, v in outputs.items()}
            deltas = {k: img.astype(np.int16) - identity for k, img in outputs.items()}
            scores = {k: float(np.abs(delta[mask]).mean()) for k, delta in deltas.items()}
            residual = deltas['full'] - deltas['features_only'] - deltas['labels_only']
            directions[name] = dict(output_sha256=hashes, bridge_mae=bridge(name, outputs['full']),
                                    raw_label_change=float((reference_labels != changed_labels).mean()),
                                    neutral_mae=scores, interaction_mae=float(np.abs(residual[mask]).mean()))
        cases.append(dict(pair=pair, identity_sha256=identity_hash, identity_bridge_mae=identity_bridge,
                          directions=directions))
        assert time.monotonic() - start < 90
    scores = [row['neutral_mae'] for case in cases for row in case['directions'].values()]
    high = sum(row['features_only'] >= .80 * row['full'] for row in scores)
    low = sum(row['features_only'] <= .80 * row['full'] and row['full'] - row['features_only'] >= .25 for row in scores)
    snow = cases[0]['directions']['background_green_100']['neutral_mae']
    assert cases[0]['pair'] == '785-3553'
    report = dict(cases=cases, retained_at_least_80=high, reduced_at_least_20_and_point25=low,
                  snow_retained=snow['features_only'] / snow['full'], seconds=time.monotonic() - start,
                  prior_sha256=digest(PRIOR), parent_sha256=PARENT_SHA, script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('cap-vstnet-replay-protocol.md')),
                  adapter_sha256=digest(Path(__file__).with_name('probe_cap_vstnet.py')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(high, low, report['snow_retained'], report['seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
