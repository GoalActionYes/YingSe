"""Keep the color grid fixed and remove chroma from only its sampling guide."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import softmax

from background_intervention import LUMA
from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json
from spatial_render import SpatialEngine, finish_spatial


def luminance_rgb(rgb):
    weights = rgb.new_tensor(LUMA).view(1, 3, 1, 1)
    mono = (rgb * weights).sum(dim=1, keepdim=True)
    gray = mono.expand_as(rgb).contiguous()
    torch.testing.assert_close((gray * weights).sum(dim=1, keepdim=True), mono, rtol=0, atol=1e-7)
    return gray


def gray_guide_input(module, args):
    if len(args) != 1:
        raise ValueError('Unexpected native guide call')
    return (luminance_rgb(args[0]),)


class GuideChromaProbe:
    def __init__(self):
        self.engine = SpatialEngine('multiref-v1')

    @lru_cache(maxsize=10)
    @torch.inference_mode()
    def guide_change(self, shape, data):
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        sampler = self.engine.model.sampler
        weights = softmax(sampler.L_weights, dim=1)
        total, peak = 0., 0.
        for start in range(0, shape[0], 256):
            original = self.engine.tensor(rgb[start:start + 256])
            a, b = [(sampler.pre_guide(value) * weights).sum(dim=1)
                    for value in (original, luminance_rgb(original))]
            delta = ((a - .5).mul(2).clamp(-1, 1) - (b - .5).mul(2).clamp(-1, 1)).abs()
            if not torch.isfinite(delta).all():
                raise ValueError('Nonfinite grid depth difference')
            total += float(delta.double().sum())
            peak = max(peak, float(delta.max()))
        return dict(mean=total / (shape[0] * shape[1]), max=peak)

    @torch.inference_mode()
    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(replay=False, reason='same_image')
        prepared = self.engine.prepare(target, reference)
        depth_delta = self.guide_change(target.shape, target.tobytes())
        with self.engine.model.sampler.pre_guide.register_forward_pre_hook(gray_guide_input):
            candidate = finish_spatial(target, prepared.apply(target), .8)
        replay = finish_spatial(target, prepared.apply(target), .8)
        np.testing.assert_array_equal(replay, baseline, err_msg='Frozen 13 output changed after removing hook')
        delta = candidate.astype(np.int16) - replay
        return candidate, dict(replay=True, depth_delta=depth_delta, rgb_delta=float(np.abs(delta).mean()),
                               changed_count=int(np.count_nonzero(np.any(delta != 0, axis=2))))


def run(out):
    started = time.monotonic()
    torch.set_num_threads(4)
    docs = ROOT / 'docs/research'
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    pair = docs / '2026-11-18-psbench-action-pair-evidence.json'
    assert digest(pair) == PSBENCH_SHA and toggle['parent_sha256'] == PARENT_SHA and parent['strength'] == .8
    out.mkdir(exist_ok=False)
    probe = GuideChromaProbe()
    assert probe.engine.checkpoint_sha == parent['model_sha256']
    sample = np.full((32, 32, 3), 120, np.uint8)
    np.testing.assert_array_equal(probe.measure(sample, sample.copy(), sample - 20)[0], sample)
    background = background_controls(out, parent, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in parent['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', pair, probe.measure)
    seconds = time.monotonic() - started
    if seconds >= 100:
        raise TimeoutError('Frozen target guide test exceeded its budget')
    source = Path(__file__)
    report = dict(background=background, background_gate=decision(background), local=local, real=real,
                  seconds=seconds, same_image=True, weight_sha256=probe.engine.checkpoint_sha,
                  parent_sha256=PARENT_SHA, toggle_sha256=PARENT_REPORT, real_sha256=PSBENCH_SHA,
                  source_sha256=digest(source), protocol_sha256=digest(source.with_name('target-guide-chroma-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report['background_gate']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
