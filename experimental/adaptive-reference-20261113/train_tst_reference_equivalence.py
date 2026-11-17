"""Matched training on author-indexed reference swaps, with or without same-result loss."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST, load
from independent_coco import digest
from probe_tst_reference_equivalence import run as evaluate
from spatial_render import SpatialEngine

SPLIT_SHA = 'b6897618cc3809fe9c3e3f27e0ebde14d6380dc4582725a1e54a4d88c52c37cb'
BASE = COHORT.parent.parent
SPLIT = BASE / 'reference-equivalence-split-min-conflict-v1'
FROZEN_BASELINE = BASE / 'reference-equivalence-development-baseline-v1/results.json'


def tensor_samples(rows):
    entries = {row['original_path']: row for row in load(MANIFEST, HASHES[2])['files']}
    cache = {}

    def get(key):
        if key not in cache:
            row = entries[key]
            path = COHORT.parent / row['prepared_path']
            assert digest(path) == row['sha256']
            with Image.open(path) as image:
                rgb = np.array(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
            cache[key] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return cache[key]

    samples = []
    for row in rows:
        target = get(row['content'])[None]
        refs = torch.stack([get(key) for key in row['refs']])
        truth_a, truth_b = (get(row[key]) for key in ('same_truth', 'different_truth'))
        samples.append((target, refs, torch.stack([truth_a, truth_a, truth_b])))
    return samples


def optimize(engine, samples, orders, weight, response_weight=0.):
    model, api = engine.model, engine.api
    parameters = list(model.hist_projection.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
    histogram = api['batch_histogram_1D']
    convert = api['srgb_to_ycbcr']
    epochs = []
    started = time.monotonic()
    for epoch, order in enumerate(orders, start=1):
        losses = []
        for index in order:
            target, refs, truth = (tensor.to(engine.device, dtype=torch.float32) / 255 for tensor in samples[index])
            optimizer.zero_grad(set_to_none=True)
            content_hist = histogram(convert(target)).expand(3, -1, -1)
            reference_hist = histogram(convert(refs))
            pred, _ = model(target.expand(3, -1, -1, -1), content_hist, reference_hist)
            pred = pred.clamp(0, 1)
            rgb_loss = F.l1_loss(pred, truth)
            agreement = F.l1_loss(pred[0], pred[1])
            loss = rgb_loss + weight * agreement
            response = None
            if response_weight:
                response = F.l1_loss(pred[2] - pred[:2], truth[2] - truth[:2])
                loss = loss + response_weight * response
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            row = [float(rgb_loss.detach()), float(agreement.detach())]
            if response is not None:
                row.append(float(response.detach()))
            losses.append(row)
            assert time.monotonic() - started <= 240
        rgb, consistency, *response_mean = np.asarray(losses).mean(axis=0).tolist()
        summary = dict(epoch=epoch, rgb_l1=rgb, same_result_l1=consistency, seconds=time.monotonic()-started)
        if response_mean:
            summary['different_result_delta_l1'] = response_mean[0]
        epochs.append(summary)
        print(weight, epochs[-1], flush=True)
    return epochs


def metrics(report):
    rows = [row['regions'] for row in report['cases']]
    return dict(passed_cases=report['passed_cases'],
                same_swap=float(np.mean([row['whole']['same_swap'] for row in rows])),
                projection=float(np.mean([row['whole']['projection'] or 0 for row in rows])),
                whole_mae=float(np.mean([row['whole']['errors'] for row in rows])),
                neutral_mae=float(np.mean([row['neutral_proxy']['errors'] for row in rows])))


def decision(baseline, control, candidate):
    a, b, c = (metrics(r) for r in (baseline, control, candidate))
    degradations = [x - y for case, original in zip(candidate['cases'], baseline['cases'], strict=True)
                    for x, y in zip(case['regions']['whole']['errors'], original['regions']['whole']['errors'], strict=True)]
    checks = dict(case_count=c['passed_cases'] >= 18 and c['passed_cases'] >= b['passed_cases'] + 3,
                  same_swap=c['same_swap'] <= .90 * min(a['same_swap'], b['same_swap']),
                  different_response=c['projection'] >= max(a['projection'], b['projection']) - .05,
                  whole_mae=c['whole_mae'] <= min(a['whole_mae'], b['whole_mae']) + .25,
                  neutral_mae=c['neutral_mae'] <= min(a['neutral_mae'], b['neutral_mae']) + .25,
                  severe_regression=max(degradations) <= 2.)
    return dict(baseline=a, control=b, candidate=c, max_regression=max(degradations), checks=checks,
                proceed=all(checks.values()))


def run(out):
    torch.set_num_threads(4)
    assignment = load(SPLIT / 'private-assignment.json', SPLIT_SHA)
    assert len(assignment['training']) == 82 and len(assignment['development']) == 24
    saved = json.loads(FROZEN_BASELINE.read_text(encoding='utf-8'))
    assert saved['metadata']['assignment_sha256'] == SPLIT_SHA and saved['passed_cases'] == 7
    engine = SpatialEngine('multiref-v1')
    out.mkdir(exist_ok=False)
    baseline = evaluate(out / 'initial', SPLIT, engine)
    assert baseline['predictions'] == saved['predictions'] and baseline['cases'] == saved['cases']
    samples = tensor_samples(assignment['training'])
    rng = np.random.default_rng(20261118)
    orders = [rng.permutation(len(samples)).tolist() for _ in range(3)]
    (out / 'order.json').write_text(json.dumps(orders), encoding='utf-8')
    hist = engine.model.hist_projection
    initial = {key: tensor.detach().cpu().clone() for key, tensor in hist.state_dict().items()}
    hist.requires_grad_(True)
    scores = {}
    arms = {}
    for name, weight in [('control', 0.), ('candidate', .25)]:
        hist.load_state_dict(initial, strict=True)
        epochs = optimize(engine, samples, orders, weight)
        checkpoint = out / f'{name}-hist-projection.pth'
        torch.save(hist.state_dict(), checkpoint)
        scores[name] = evaluate(out / name, SPLIT, engine)
        arms[name] = dict(epochs=epochs, weights_sha256=digest(checkpoint),
                          results_sha256=digest(out / name / 'results.json'))
    verdict = decision(baseline, scores['control'], scores['candidate'])
    report = dict(assignment_sha256=SPLIT_SHA, frozen_baseline_sha256=digest(FROZEN_BASELINE),
                  init_checkpoint_sha256=engine.checkpoint_sha, initial_exact_outputs=72,
                  order_sha256=digest(out / 'order.json'), training_groups=len(samples), arms=arms, decision=verdict,
                  source_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name('tst-reference-equivalence-training-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(verdict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
