"""Compare person compositing with a frozen foreground harmonization model."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from independent_coco import digest
from person_segmenter import CHECKPOINT_SHA, PersonSegmenter
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json


VENDOR = ROOT / '.tools/harmonizer-eccv2022-20261118'
REVISION = '48ecd70becbff50ccaf576db0e64212dbc494e26'
WEIGHT_SHA = '401dd03706fc6fa37b4600ae4c643a2fa9ad0f11351acfbf030cb5d3490de6ca'
PROFILES = ('direct', 'harmonizer')


def get_model():
    report = json.loads((VENDOR / 'source-manifest.json').read_text(encoding='utf-8'))
    if report['commit'] != REVISION or len(report['files']) != 12:
        raise ValueError('Unexpected upstream source manifest')
    for row in report['files']:
        if digest(VENDOR / row['path']) != row['sha256']:
            raise ValueError('Upstream source changed')
    weight = VENDOR / 'harmonizer.pth'
    if digest(weight) != WEIGHT_SHA:
        raise ValueError('Unexpected Harmonizer checkpoint')
    sys.path.extend((str(VENDOR), str(VENDOR / 'deps')))
    from src.model import Harmonizer
    model = Harmonizer()
    model.load_state_dict(torch.load(weight, weights_only=True, map_location='cpu'), strict=True)
    return model.to('cuda').eval().requires_grad_(False)


class Probe:
    def __init__(self):
        self.segmenter = PersonSegmenter('cuda')
        self.model = get_model()
        self.profile = None

    @lru_cache(maxsize=10)
    def probability(self, shape, data):
        target = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        return self.segmenter.predict(target)[0]

    @torch.inference_mode()
    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(applied=False, reason='same_image')
        p = self.probability(target.shape, target.tobytes())
        coverage = float(p.mean())
        if coverage < .01:
            return baseline.copy(), dict(applied=False, reason='no_person', coverage=coverage)
        mask = torch.from_numpy(p)[None, None].to('cuda')
        original = torch.from_numpy(target.copy()).permute(2, 0, 1)[None].to('cuda', torch.float32) / 255
        edited = torch.from_numpy(baseline.copy()).permute(2, 0, 1)[None].to('cuda', torch.float32) / 255
        composite = original * mask + edited * (1 - mask)
        stats = dict(applied=True, coverage=coverage)
        prediction = composite
        if self.profile == 'harmonizer':
            arguments = self.model.predict_arguments(composite, mask)
            prediction = self.model.restore_image(composite, mask, arguments)[-1]
            stats['arguments'] = [float(x.item()) for x in arguments]
        if not torch.isfinite(prediction).all() or prediction.shape != original.shape:
            raise ValueError('Invalid harmonized result')
        rgb = (prediction[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        np.testing.assert_array_equal(rgb[p == 0], baseline[p == 0])
        return rgb, stats


def run(out):
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    docs = ROOT / 'docs/research'
    manifest = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    psbench = docs / '2026-11-18-psbench-action-pair-evidence.json'
    segmenter = ROOT / '.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    assert toggle['parent_sha256'] == PARENT_SHA and manifest['strength'] == .8
    assert digest(psbench) == PSBENCH_SHA and digest(segmenter) == CHECKPOINT_SHA
    out.mkdir(exist_ok=False)
    probe = Probe()
    arms = {}
    for profile in PROFILES:
        folder = out / profile
        folder.mkdir()
        probe.profile = profile
        sample = np.full((32, 32, 3), 127, dtype=np.uint8)
        present, _ = probe.measure(sample, sample.copy(), sample - 20)
        np.testing.assert_array_equal(present, sample)
        background = background_controls(folder, manifest, probe.measure)
        local = local_controls(probe, folder, toggle, {r['pair']: r for r in manifest['cases']})
        real = authored(folder, BASE / 'psbench-shared-local-v1', psbench, probe.measure)
        arms[profile] = dict(background=background, background_gate=decision(background), local=local, real=real)
    seconds = time.monotonic() - start
    if seconds > 150:
        raise TimeoutError('Frozen harmonization budget exceeded')
    source = Path(__file__)
    report = dict(arms=arms, seconds=seconds, same_image=True,
                  weight_sha256=WEIGHT_SHA, segmentation_sha256=CHECKPOINT_SHA,
                  vendor_manifest_sha256=digest(VENDOR / 'source-manifest.json'),
                  protocol_sha256=digest(source.with_name('person-harmonizer-protocol.md')),
                  source_sha256=digest(source), psbench_sha256=PSBENCH_SHA,
                  toggle_sha256=PARENT_REPORT, background_sha256=PARENT_SHA)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: group['background_gate'] for name, group in arms.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
