"""Reproducible representational probe, not a semantic transfer algorithm.

Run from repository root with its Python dependencies. Results stay in outputs.
Identical RGB patches occur in both regions; desired edits differ by region.
This deliberately constructed test proves a global-map limit, not user quality.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reference_color_app.pipeline import analyze, algorithm_fingerprint
from reference_color_app.lut import apply_mapping
from regional import fit, apply


def main():
    output = ROOT / 'outputs/adaptive-reference-20261113/global-limit'
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1311)
    patch = rng.integers(40, 211, size=(96, 96, 3), dtype=np.uint8)
    source = np.concatenate([patch, patch], axis=1)
    x = patch.astype(np.float64)
    gray = (x @ np.array([.299, .587, .114]))[..., None]
    cases = {
        'identity': (x, x),
        'uniform_brightness': (x + 15, x + 15),
        'regional_brightness': (x + 24, x - 24),
        'regional_contrast': ((x - 128) * 1.3 + 128, (x - 128) * .7 + 128),
        'regional_saturation': (gray + (x - gray) * 1.4, gray + (x - gray) * .4),
        'regional_color_balance': (x + [-18, 0, 18], x + [18, 0, -18]),
    }
    Image.fromarray(source).save(output / 'target.png')
    rows = []
    for name, (a, b) in cases.items():
        desired = np.rint(np.clip(np.concatenate([a, b], axis=1), 0, 255)).astype(np.uint8)
        reference = output / (name + '-reference.png')
        Image.fromarray(desired).save(reference)
        start = time.perf_counter()
        mapping = analyze(output / 'target.png', reference)
        seconds = time.perf_counter() - start
        result = apply_mapping(source, mapping, 1.)
        Image.fromarray(result).save(output / (name + '-baseline.png'))
        masks = np.zeros((*source.shape[:2], 2), np.float32)
        masks[:, :96, 0] = 1
        masks[:, 96:, 1] = 1
        regional_model = fit(result, desired, masks, masks)
        regional = apply(result, masks, regional_model)
        Image.fromarray(regional).save(output / (name + '-oracle-regions.png'))
        # For identical inputs a global deterministic map must return the same
        # output. Triangle inequality gives this exact minimum L1 error bound.
        left, right = desired[:, :96].astype(float), desired[:, 96:].astype(float)
        bound = float(np.abs(left - right).mean() / 2)
        error = float(np.abs(result.astype(float) - desired).mean())
        assert np.array_equal(result[:, :96], result[:, 96:])
        assert error + 1e-6 >= bound
        rows.append(dict(case=name, baseline_rgb_mae=error,
                         oracle_regions_candidate_mae=float(np.abs(regional.astype(float)-desired).mean()),
                         oracle_warning='Known left/right masks supplied; not automatic semantic evidence',
                         any_global_rgb_map_minimum_mae=bound,
                         analyze_seconds=seconds,
                         aligned_fit=mapping.diagnostics.get('aligned_reference'),
                         repeated_input_outputs_identical=True))
    report = dict(version=1, seed=1311, algorithm_fingerprint=algorithm_fingerprint(),
                  scope='Synthetic repeated-color regions; not natural quality or semantic recognition evidence',
                  timing_scope='Small 192x96 RGB inputs; not interactive or video performance', cases=rows)
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
