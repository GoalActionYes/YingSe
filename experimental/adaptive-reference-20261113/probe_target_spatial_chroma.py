"""Remove target chroma from only the frozen spatial encoder input."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from background_intervention import LUMA
from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from training_checks import finite_output


class SpatialChromaProbe:
    def __init__(self):
        self.engine = SpatialEngine('multiref-v1')

    @torch.inference_mode()
    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(replay=False, reason='same_image')
        engine = self.engine
        api = engine.api
        rgb, look = engine.analysis(target), engine.analysis(reference)
        histogram = api['batch_histogram_1D']
        convert = api['srgb_to_ycbcr']
        condition = torch.cat((histogram(convert(rgb)), histogram(convert(look))), dim=1)
        low = api['TF'].resize(rgb, (64, 64), antialias=True)
        weights = low.new_tensor(LUMA).view(1, 3, 1, 1)
        y = (low * weights).sum(dim=1, keepdim=True)
        gray = y.expand_as(low).contiguous()
        torch.testing.assert_close((gray * weights).sum(dim=1, keepdim=True), y, rtol=0, atol=1e-7)
        grids = [finite_output(api['lowpass'](engine.model.apply_grid(x, condition)), (1, 13, 8, 16, 16))
                 for x in (low, gray)]
        replay, output = [finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8) for grid in grids]
        np.testing.assert_array_equal(replay, baseline, err_msg='Frozen 13 output changed')
        delta = output.astype(np.int16) - replay
        return output, dict(replay=True, input_chroma_change=float(torch.abs(low - gray).mean()),
                            grid_delta=float(torch.abs(grids[0] - grids[1]).mean()),
                            rgb_delta=float(np.abs(delta).mean()),
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
    probe = SpatialChromaProbe()
    assert probe.engine.checkpoint_sha == parent['model_sha256']
    example = np.full((32, 32, 3), 122, np.uint8)
    np.testing.assert_array_equal(probe.measure(example, example.copy(), example - 20)[0], example)
    background = background_controls(out, parent, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in parent['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', pair, probe.measure)
    seconds = time.monotonic() - started
    if seconds >= 100:
        raise TimeoutError('Frozen spatial chroma budget exceeded')
    file = Path(__file__)
    result = dict(background=background, background_gate=decision(background), local=local, real=real,
                  seconds=seconds, same_image=True, weight_sha256=probe.engine.checkpoint_sha,
                  parent_sha256=PARENT_SHA, toggle_sha256=PARENT_REPORT, real_sha256=PSBENCH_SHA,
                  source_sha256=digest(file), protocol_sha256=digest(file.with_name('target-spatial-chroma-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['background_gate']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
