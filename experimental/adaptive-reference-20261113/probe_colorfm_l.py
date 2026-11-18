"""Screen the author's ColorFM-L model on frozen two-image development inputs."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_neutral_relation_balance import PARENT_REPORT, PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json


SOURCE_SHA = '90b5086dd323955bde9d4a12f029b67c1d0a372179653b984dd58922c4d48a63'
WEIGHT_SHA = 'e31eccb7862353f69d043590300a1675bd274654a8ab16f9d662a966ea91b33f'


class ColorFMProbe:
    def __init__(self, upstream):
        upstream = upstream.resolve()
        sources = checked_json(upstream / 'verified-source.json', SOURCE_SHA)
        for row in sources:
            if digest(upstream / row['path']) != row['sha256']:
                raise ValueError(f"Unexpected author source: {row['path']}")
        weights = upstream / 'ColorFM_L.pth'
        if weights.stat().st_size != 123249227 or digest(weights) != WEIGHT_SHA:
            raise ValueError('Unexpected ColorFM-L weight identity')
        path = upstream / 'models/colorfm_l.py'
        original = path.read_text(encoding='utf-8')
        rank_warning = 'from lightning.pytorch.utilities.rank_zero import rank_zero_warn'
        assert original.count(rank_warning) == 1
        assert 'models' not in sys.modules, 'A different top-level models package is active'
        sys.path.insert(0, str(upstream))
        # The author calls this only to log that xFormers is unavailable.
        module = {'__name__': 'colorfm_l_screen', '__file__': str(path)}
        exec(compile(original.replace(rank_warning, 'from warnings import warn as rank_zero_warn'),
                     str(path), 'exec'), module)
        model = module['ColorFM_L'](cfg=None)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.model = model.cuda().eval()

    @torch.inference_mode()
    def transfer(self, target, reference):
        def tensor(image):
            value = np.ascontiguousarray(image).astype(np.float32) / 255
            return torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).cuda()

        result, _, _ = self.model(tensor(target), tensor(reference))
        raw = result[0].permute(1, 2, 0).cpu().numpy()
        assert raw.shape == target.shape and np.isfinite(raw).all()
        original = target.astype(np.float32) / 255
        return np.rint(np.clip(original + (raw - original) * .8, 0, 1) * 255).astype(np.uint8)

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(reason='same_image')
        return self.transfer(target, reference), dict(reason='colorfm_l')


def run(upstream, out):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261118)
    docs = ROOT / 'docs/research'
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    pair = docs / '2026-11-18-psbench-action-pair-evidence.json'
    assert digest(pair) == PSBENCH_SHA and toggle['parent_sha256'] == PARENT_SHA and parent['strength'] == .8
    probe = ColorFMProbe(upstream)
    sample = np.tile(np.array([90, 120, 150], np.uint8), (32, 32, 1))
    sample[:, :12] = 70
    native = probe.transfer(sample, sample)
    same = probe.measure(sample, sample, 255-sample)[0]
    assert np.array_equal(same, sample)
    assert np.array_equal(native[1, 16], native[-2, -2])
    out.mkdir()
    background = background_controls(out, parent, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in parent['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', pair, probe.measure)
    if time.monotonic() - started >= 300:
        raise TimeoutError('ColorFM-L run exceeded budget')
    file = Path(__file__)
    report = dict(background=background, background_gate=decision(background), local=local, real=real,
                  same_image=True, native_same_image_mae=float(np.abs(native.astype(np.int16)-sample).mean()),
                  weight_sha256=WEIGHT_SHA, author_source_sha256=SOURCE_SHA, parent_sha256=PARENT_SHA,
                  toggle_sha256=PARENT_REPORT, psbench_sha256=PSBENCH_SHA, source_sha256=digest(file),
                  protocol_sha256=digest(file.with_name('colorfm-l-screen-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(background=report['background_gate'], local={k: v['gates'] for k,v in local.items()},
                          real={k: v['gate'] for k,v in real.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
