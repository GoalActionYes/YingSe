"""Match RGB support on coordinates unchanged by the cold-color axis."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import LUMA
from independent_coco import digest
from person_segmenter import CHECKPOINT_SHA, PersonSegmenter
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import COOL
from probe_local_shadow_intercept import small
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json


AXIS = COOL / np.linalg.norm(COOL)
ORTHOGONAL = np.cross(np.ones(3), AXIS)
ORTHOGONAL /= np.linalg.norm(ORTHOGONAL)
BIN_COUNT = 24


def bins(rgb):
    values = rgb.astype(float)
    intensity = np.floor((values @ LUMA)/32).astype(int)
    other = np.floor((values @ ORTHOGONAL + 288)/24).astype(int)
    assert ((other >= 0) & (other < BIN_COUNT)).all()
    return intensity*BIN_COUNT + other


def summary(support, measured, probability):
    ids = bins(support)
    value = measured.astype(float) @ AXIS
    result = []
    for mask in (probability >= .8, probability <= .2):
        keys = ids[mask]
        counts = np.bincount(keys, minlength=8*BIN_COUNT)
        sums = np.bincount(keys, weights=value[mask], minlength=8*BIN_COUNT)
        result.append((counts, sums))
    return result


def corrections(target, reference):
    offsets, reused = [], []
    for (count, total), (other_count, other_total) in zip(target, reference, strict=True):
        common = (count >= 96) & (other_count >= 96)
        reused.append(int(common.sum()))
        if reused[-1] >= 3:
            difference = other_total[common]/other_count[common] - total[common]/count[common]
            offsets.append(float(np.median(difference)))
    return offsets, reused


class SupportProbe:
    def __init__(self):
        self.segmenter = PersonSegmenter('cuda')
        self.targets = {}

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(applied=False, reason='same_image')
        key = target.shape, target.tobytes()
        if key not in self.targets:
            self.targets[key] = self.segmenter.predict(target)[0]
        probability = self.targets[key]
        reduced = np.asarray(Image.fromarray(probability).resize((256, 256), Image.Resampling.BILINEAR))
        ref = small(reference)
        ref_probability = self.segmenter.predict(ref)[0]
        offsets, cells = corrections(summary(small(target), small(baseline), reduced),
                                     summary(ref, ref, ref_probability))
        if len(offsets) != 2:
            return baseline.copy(), dict(applied=False, shared_cells=cells, reason='support')
        limit = 8/np.max(np.abs(AXIS))
        subject, environment = np.clip(.5*np.array(offsets), -limit, limit)
        weight = probability.astype(float)*subject + (1-probability.astype(float))*environment
        change = weight[..., None]*AXIS
        np.testing.assert_allclose(change @ LUMA, 0, atol=1e-12)
        raw = baseline + change
        refined = np.rint(raw.clip(0, 255)).astype(np.uint8)
        return refined, dict(applied=True, shared_cells=cells, offsets=offsets,
                             axis_amount=[float(subject), float(environment)],
                             clip_fraction=float(((raw < 0) | (raw > 255)).mean()))


def synthetic():
    sample = np.tile(np.array([54, 92, 112, 151, 185], dtype=float), (32, 32, 1)).reshape(32, 160)
    rgb = np.repeat(sample[..., None], 3, axis=-1)
    person = np.broadcast_to(np.arange(32)[:, None] < 16, sample.shape)
    shifted = rgb + 7*AXIS
    np.testing.assert_array_equal(bins(rgb), bins(shifted))
    offset, number = corrections(summary(rgb, rgb, person), summary(shifted, shifted, person))
    assert all(cell >= 3 for cell in number)
    np.testing.assert_allclose(offset, [7, 7], atol=1e-11)


def run(out):
    started = time.monotonic()
    synthetic()
    torch.set_num_threads(4)
    docs = ROOT/'docs/research'
    toggle = checked_json(docs/'2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    manifest = checked_json(PARENT/'results.json', PARENT_SHA)
    assert toggle['parent_sha256'] == PARENT_SHA and manifest['strength'] == .8
    psbench = docs/'2026-11-18-psbench-action-pair-evidence.json'
    assert digest(psbench) == PSBENCH_SHA
    checkpoint = ROOT/'.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    assert digest(checkpoint) == CHECKPOINT_SHA
    out.mkdir()
    probe = SupportProbe()
    flat = np.full((32, 32, 3), 127, np.uint8)
    kept, _ = probe.measure(flat, flat.copy(), flat-10)
    np.testing.assert_array_equal(flat, kept)
    background = background_controls(out, manifest, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in manifest['cases']})
    real = authored(out, BASE/'psbench-shared-local-v1', psbench, probe.measure)
    assert time.monotonic()-started <= 90
    report = dict(background=background, background_gate=decision(background), local=local, real=real,
                  same_image=True, synthetic=True, source_sha256=digest(Path(__file__)),
                  background_sha256=PARENT_SHA, toggle_sha256=PARENT_REPORT, real_sha256=PSBENCH_SHA,
                  segmentation_sha256=CHECKPOINT_SHA)
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['background_gate']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
