"""Require distinct subject and background evidence for a shared tint adjustment."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from independent_coco import digest
from person_segmenter import CHECKPOINT_SHA, PersonSegmenter
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_reference_tint_envelope import Probe, correction, supports
from probe_roundtrip_grade_choice import checked_json


PREVIOUS = BASE / 'reference-tint-envelope-v1'
PREVIOUS_SHA = 'b83d9e22bb17bd8a3a2740201ede6b2d1aa3e3c5c7c7e0e482d75d31df6d94cc'


def envelope(blocks, center):
    first, third = np.quantile(blocks, [.25, .75])
    return float(min(-3, third - center - 3)), float(max(3, first - center + 3))


def shared_envelope(fg, bg, center):
    first, second = envelope(fg, center), envelope(bg, center)
    return max(first[0], second[0]), min(first[1], second[1])


class SemanticProbe(Probe):
    def __init__(self):
        super().__init__('reference')
        self.segmenter = PersonSegmenter('cuda')

    @lru_cache(maxsize=16)
    def reference_regions(self, shape, data):
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        probability, _ = self.segmenter.predict(rgb)
        fg = probability >= .8
        bg = probability <= .2
        values = [supports(np.where(region[..., None], rgb, 0)) for region in (fg, bg)]
        return values, [float(region.mean()) for region in (fg, bg)]

    def measure(self, target, reference, baseline):
        candidate, previous = super().measure(target, reference, baseline)
        if 'limits' not in previous:
            return candidate, previous
        (fg, bg), coverage = self.reference_regions(reference.shape, reference.tobytes())
        metadata = dict(region_blocks=[len(fg), len(bg)], region_coverage=coverage,
                        old_limits=previous['limits'])
        if len(fg) < 2 or len(bg) < 4:
            return candidate, dict(**previous, **metadata, semantic=False, reason='region_support')
        target_blocks = self.target_stats(target.shape, target.tobytes())
        limits = shared_envelope(fg, bg, np.median(target_blocks))
        corrected, diagnostics = correction(target, baseline, limits)
        return corrected, dict(**diagnostics, **metadata, semantic=True,
                               limits=list(limits), blocks=previous['blocks'])


def run(out):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert shared_envelope([0, 0], [30] * 4, 0.) == (-3., 3.)
    assert shared_envelope([12, 12], [30] * 4, 0.) == (-3., 15.)
    assert shared_envelope([-12, -12], [-30] * 4, 0.) == (-15., 3.)
    prior = checked_json(PREVIOUS / 'results.json', PREVIOUS_SHA)
    docs = ROOT / 'docs/research'
    frozen = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    psbench = docs / '2026-11-18-psbench-action-pair-evidence.json'
    checkpoint = ROOT / '.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    assert prior['background_sha256'] == toggle['parent_sha256'] == PARENT_SHA and frozen['strength'] == .8
    assert digest(psbench) == PSBENCH_SHA and digest(checkpoint) == CHECKPOINT_SHA
    out.mkdir(exist_ok=False)
    probe = SemanticProbe()
    example = np.full((32, 32, 3), 127, np.uint8)
    np.testing.assert_array_equal(probe.measure(example, example.copy(), example - 20)[0], example)
    background = background_controls(out, frozen, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in frozen['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', psbench, probe.measure)
    seconds = time.monotonic() - started
    if seconds > 100:
        raise TimeoutError('Frozen semantic tint screening budget exceeded')
    file = Path(__file__)
    result = dict(background=background, background_gate=decision(background), local=local, real=real,
                  seconds=seconds, same_image=True, previous_sha256=PREVIOUS_SHA, parent_sha256=PARENT_SHA,
                  toggle_sha256=PARENT_REPORT, real_sha256=PSBENCH_SHA, segmentation_sha256=CHECKPOINT_SHA,
                  source_sha256=digest(file), protocol_sha256=digest(file.with_name('semantic-tint-support-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['background_gate']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
