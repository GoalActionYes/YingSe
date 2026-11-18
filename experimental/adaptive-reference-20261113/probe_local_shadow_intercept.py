"""Test local shade-line intercepts as evidence for regional color correction."""
import argparse
import json
from functools import lru_cache
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
from probe_neutral_relation_balance import PSBENCH_SHA, balanced_render, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision
from probe_roundtrip_grade_choice import checked_json


def blocks(image):
    return image.reshape(16, 16, 16, 16, *image.shape[2:]).swapaxes(1, 2).reshape(
        256, 256, *image.shape[2:]
    )


def small(rgb):
    if rgb.shape[:2] == (256, 256):
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))


def estimate(support, measured, probability):
    good = blocks(((support >= 8) & (support <= 247)).all(axis=-1))
    origin_y = blocks(support.astype(float) @ LUMA)
    rgb = blocks(measured.astype(float))
    intensity = rgb @ LUMA
    chroma = rgb - intensity[..., None]
    p = blocks(probability)
    vectors, support_counts = [], []
    for region in (p >= .8, p <= .2):
        valid = good & region
        fits = []
        for i in np.flatnonzero(valid.sum(axis=1) >= 192):
            selected = valid[i]
            if origin_y[i, selected].std() < 6:
                continue
            y = intensity[i, selected]
            normalized = y - y.mean()
            spread = float(normalized @ normalized)
            if spread <= 1e-9:
                continue
            c = chroma[i, selected]
            slope = (normalized @ c) / spread
            fits.append(c.mean(axis=0) - y.mean() * slope)
        support_counts.append(len(fits))
        if len(fits) >= 4:
            vectors.append(np.median(fits, axis=0))
    relation = vectors[0] - vectors[1] if len(vectors) == 2 else None
    return relation, support_counts


class InterceptProbe:
    def __init__(self):
        self.segmenter = PersonSegmenter('cuda')

    @lru_cache(maxsize=10)
    def target_probability(self, shape, data):
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        return self.segmenter.predict(rgb)[0]

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(applied=False, reason='same_image')
        probability = self.target_probability(target.shape, target.tobytes())
        if probability.mean() < .01:
            return baseline.copy(), dict(applied=False, reason='target_person')
        p = np.asarray(Image.fromarray(probability).resize((256, 256), Image.Resampling.BILINEAR))
        measured, target_count = estimate(small(target), small(baseline), p)
        stats = dict(target_blocks=target_count)
        if measured is None:
            return baseline.copy(), dict(applied=False, reason='target_blocks', **stats)
        ref = small(reference)
        ref_prob = self.segmenter.predict(ref)[0]
        expected, ref_count = estimate(ref, ref, ref_prob)
        stats['reference_blocks'] = ref_count
        if expected is None:
            return baseline.copy(), dict(applied=False, reason='reference_blocks', **stats)
        rendered, diagnostics = balanced_render(baseline, probability, expected - measured)
        return rendered, dict(applied=True, observed=measured.tolist(), desired=expected.tolist(),
                              **stats, **diagnostics)


def synthetic_check():
    shades = np.tile(np.linspace(36, 156, 256).reshape(16, 16), (16, 16))
    person = np.indices(shades.shape)[1] < 128
    _, columns = np.indices(shades.shape)
    slopes = np.array([[1.12, .94, .84], [.90, 1.04, 1.12], [1.07, 1., .75], [.84, 1.02, 1.10]])
    offset = np.array([-7., 1., 14.])
    offset -= offset @ LUMA
    reflectance = shades[..., None] * slopes[(columns // 16) % len(slopes)]
    rhs = reflectance + np.where(person[..., None], offset, 0)
    found, counts = estimate(reflectance, rhs, person)
    assert counts == [128, 128]
    np.testing.assert_allclose(found, offset, atol=1e-10, rtol=0)
    zero, _ = estimate(reflectance, reflectance * 1.1, person)
    np.testing.assert_allclose(zero, 0, atol=1e-10, rtol=0)


def run(out):
    start = time.monotonic()
    synthetic_check()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    docs = ROOT / 'docs/research'
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    manifest = checked_json(PARENT / 'results.json', PARENT_SHA)
    assert toggle['parent_sha256'] == PARENT_SHA and manifest['strength'] == .8
    psbench = docs / '2026-11-18-psbench-action-pair-evidence.json'
    assert digest(psbench) == PSBENCH_SHA
    checkpoint = ROOT / '.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    assert digest(checkpoint) == CHECKPOINT_SHA
    out.mkdir(exist_ok=False)
    probe = InterceptProbe()
    sample = np.full((32, 32, 3), 127, np.uint8)
    corrected, _ = probe.measure(sample, sample.copy(), sample - 10)
    np.testing.assert_array_equal(corrected, sample)
    background = background_controls(out, manifest, probe.measure)
    local = local_controls(probe, out, toggle, {r['pair']: r for r in manifest['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', psbench, probe.measure)
    elapsed = time.monotonic() - start
    if elapsed > 90:
        raise TimeoutError('Frozen shadow intercept budget exceeded')
    file = Path(__file__)
    data = dict(background=background, background_gate=decision(background), local=local, real=real,
                seconds=elapsed, same_image=True, synthetic=True, source_sha256=digest(file),
                protocol_sha256=digest(file.with_name('local-shadow-intercept-protocol.md')),
                background_sha256=PARENT_SHA, toggle_sha256=PARENT_REPORT, real_sha256=PSBENCH_SHA,
                segmentation_sha256=CHECKPOINT_SHA)
    (out / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
