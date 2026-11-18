"""Screen a frozen RGB cycle ensemble with normal two-image inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_neutral_relation_balance import PARENT_REPORT, PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json
from spatial_render import SpatialEngine, finish_spatial


class CycleProbe:
    def __init__(self):
        self.engine = SpatialEngine('multiref-v1')

    @torch.inference_mode()
    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(reason='same_image')
        restored = []
        for shift in (1, 2):
            content = np.roll(target, shift, axis=-1)
            look = np.roll(reference, shift, axis=-1)
            output = finish_spatial(content, self.engine.prepare(content, look).apply(content), .8)
            restored.append(np.roll(output, -shift, axis=-1))
        combined = np.rint((baseline.astype(np.float32) + restored[0] + restored[1]) / 3).astype(np.uint8)
        return combined, dict(
            reason='rgb_cycle',
            mean_change=float(np.abs(combined.astype(np.int16) - baseline).mean()),
        )


def run(out):
    started = time.monotonic()
    torch.set_num_threads(4)
    docs = ROOT / 'docs/research'
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    pair = docs / '2026-11-18-psbench-action-pair-evidence.json'
    assert digest(pair) == PSBENCH_SHA and toggle['parent_sha256'] == PARENT_SHA and parent['strength'] == .8
    out.mkdir()
    probe = CycleProbe()
    assert probe.engine.checkpoint_sha == parent['model_sha256']
    sample = np.tile(np.array([90, 120, 150], np.uint8), (32, 32, 1))
    np.testing.assert_array_equal(probe.measure(sample, sample.copy(), 255 - sample)[0], sample)
    background = background_controls(out, parent, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in parent['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', pair, probe.measure)
    if time.monotonic() - started >= 180:
        raise TimeoutError('RGB cycle inference exceeded budget')
    source = Path(__file__)
    report = dict(background=background, background_gate=decision(background), local=local, real=real,
                  same_image=True, weight_sha256=probe.engine.checkpoint_sha, parent_sha256=PARENT_SHA,
                  toggle_sha256=PARENT_REPORT, psbench_sha256=PSBENCH_SHA, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('rgb-cycle-ensemble-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(background=report['background_gate'], local={k: v['gates'] for k, v in local.items()},
                          real={k: v['gate'] for k, v in real.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
