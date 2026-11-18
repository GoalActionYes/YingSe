"""Test reference-supported green/magenta limits without changing warmth or luma."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from background_intervention import LUMA
from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json


TINT = np.array([.5, -1., .5])
DIRECTION = np.array([LUMA[1], -LUMA[0] - LUMA[2], LUMA[1]])
MODES = ('fixed', 'reference')


def supports(rgb):
    x = rgb.astype(np.float64)
    y = x @ LUMA
    valid = ((y >= 40) & (y <= 220) & (np.abs(x[..., 0] - x[..., 2]) <= 40)
             & ((x > 0) & (x < 255)).all(axis=2))
    rows = np.array_split(np.arange(x.shape[0]), 4)
    cols = np.array_split(np.arange(x.shape[1]), 4)
    tint = x @ TINT
    values = []
    for r in rows:
        for c in cols:
            box = np.ix_(r, c)
            mask = valid[box]
            if mask.sum() >= 32:
                values.append(float(np.median(tint[box][mask])))
    return values


def correction(target, baseline, limits):
    rgb = target.astype(np.float64)
    rendered = baseline.astype(np.float64)
    y = rgb @ LUMA
    warm = np.abs(rgb[..., 0] - rgb[..., 2])
    weight = (np.clip((40 - warm) / 20, 0, 1) * np.clip((y - 24) / 24, 0, 1)
              * np.clip((232 - y) / 28, 0, 1))
    drift = (rendered - rgb) @ TINT
    change = ((drift.clip(*limits) - drift) * weight)[..., None] * DIRECTION
    bound = np.where(change < 0, rendered, 255 - rendered)
    scale = np.divide(bound, np.abs(change), out=np.ones_like(change), where=change != 0).min(axis=2).clip(0, 1)
    scaled = change * scale[..., None]
    np.testing.assert_allclose(scaled @ LUMA, 0, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(scaled[..., 0], scaled[..., 2])
    result = np.rint((rendered + scaled).clip(0, 255)).astype(np.uint8)
    effective = result.astype(np.float64) - rendered
    return result, dict(applied=bool(np.any(effective)),
                        adjusted_fraction=float((np.abs(change).max(axis=2) > 0).mean()),
                        gamut_fraction=float((scale < 1).mean()),
                        max_luma_error=float(np.abs(effective @ LUMA).max()),
                        max_warmth_error=float(np.abs(effective[..., 0] - effective[..., 2]).max()))


class Probe:
    def __init__(self, mode):
        self.mode = mode

    @lru_cache(maxsize=10)
    def target_stats(self, shape, data):
        return supports(np.frombuffer(data, dtype=np.uint8).reshape(shape))

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(applied=False, reason='same_image')
        original, reference_blocks = self.target_stats(target.shape, target.tobytes()), supports(reference)
        counts = [len(original), len(reference_blocks)]
        if min(counts) < 8:
            return baseline.copy(), dict(applied=False, reason='insufficient_blocks', blocks=counts)
        lower, upper = -3., 3.
        if self.mode == 'reference':
            center = np.median(original)
            first, third = np.quantile(reference_blocks, [.25, .75])
            lower, upper = float(min(lower, third - center - 3)), float(max(upper, first - center + 3))
        output, diagnostics = correction(target, baseline, (lower, upper))
        return output, dict(limits=[lower, upper], blocks=counts, **diagnostics)


def synthetic_check():
    target = np.full((32, 32, 3), 128, dtype=np.uint8)
    colored = (target.astype(float) + 20 * DIRECTION).round().astype(np.uint8)
    fixed, details = Probe('fixed').measure(target, colored, colored)
    adaptive, _ = Probe('reference').measure(target, colored, colored)
    assert details['applied'] and not details['max_warmth_error']
    assert float((fixed.astype(float) - target)[0, 0] @ TINT) <= 4
    np.testing.assert_array_equal(adaptive, colored)
    same, _ = Probe('reference').measure(target, target.copy(), colored)
    np.testing.assert_array_equal(same, target)
    more_warmth = np.tile(np.array([160, 128, 96], np.uint8), (32, 32, 1))
    no_neutral, info = Probe('reference').measure(target, more_warmth, colored)
    assert info['reason'] == 'insufficient_blocks'
    np.testing.assert_array_equal(no_neutral, colored)


def run(out):
    since = time.monotonic()
    synthetic_check()
    doc = ROOT / 'docs/research'
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(doc / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    pair = doc / '2026-11-18-psbench-action-pair-evidence.json'
    assert digest(pair) == PSBENCH_SHA and toggle['parent_sha256'] == PARENT_SHA and parent['strength'] == .8
    out.mkdir(exist_ok=False)
    arms = {}
    for name in MODES:
        folder = out / name
        folder.mkdir()
        probe = Probe(name)
        background = background_controls(folder, parent, probe.measure)
        local = local_controls(probe, folder, toggle, {row['pair']: row for row in parent['cases']})
        real = authored(folder, BASE / 'psbench-shared-local-v1', pair, probe.measure)
        arms[name] = dict(background_gate=decision(background), background=background, local=local, real=real)
    seconds = time.monotonic() - since
    if seconds > 100:
        raise TimeoutError('Frozen tint screening budget exceeded')
    script = Path(__file__)
    report = dict(arms=arms, seconds=seconds, synthetic_checks=True, background_sha256=PARENT_SHA,
                  toggle_sha256=PARENT_REPORT, psbench_sha256=PSBENCH_SHA,
                  script_sha256=digest(script), protocol_sha256=digest(script.with_name('reference-tint-envelope-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: arm['background_gate'] for name, arm in arms.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
