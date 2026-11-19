"""Check whether nonlinear global distribution matching helps portrait and style gates."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_frozen_candidate_development_crosscheck import (
    BASE, CHECKPOINT, COUNTS, MANIFESTS, STRENGTH, background, chart, facd, film_b, json_to, verify,
)
from spatial_render import finish_spatial

RNG = np.random.default_rng(61)
BASES = (np.eye(3), *(np.linalg.qr(RNG.normal(size=(3, 3)))[0] for _ in range(7)))
AXES = np.column_stack(BASES)
Q = np.linspace(.05, .95, 19)


def thumb(data):
    image = Image.fromarray(data).resize((96, 96), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255


def matrix_power(m, p):
    w, v = np.linalg.eigh(m)
    return (v * np.maximum(w, 1e-12)**p) @ v.T


def linear(a, b, full):
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    s = np.cov(a, rowvar=False) + 1e-5 * np.eye(3)
    t = np.cov(b, rowvar=False) + 1e-5 * np.eye(3)
    r = matrix_power(s, .5)
    ri = matrix_power(s, -.5)
    m = ri @ matrix_power(r @ t @ r, .5) @ ri
    return tuple((pixels - mean_a) @ m + mean_b for pixels in (a, full))


def project(a, b, full):
    for basis in BASES:
        ap, bp, fp = (value @ basis for value in (a, b, full))
        for j in range(3):
            values, counts = np.unique(ap[:, j], return_counts=True)
            sorted_b = np.sort(bp[:, j])
            ranks = counts.cumsum() - (counts + 1) * .5
            mapped = np.interp(ranks, np.arange(len(sorted_b)), sorted_b)
            ap[:, j] = np.interp(ap[:, j], values, mapped)
            fp[:, j] = np.interp(fp[:, j], values, mapped)
        a, full = ap @ basis.T, fp @ basis.T
    return a, full


def distance(a, b):
    q1, q2 = (np.quantile(pixels @ AXES, Q, axis=0) for pixels in (a, b))
    return float(np.sqrt(np.mean(np.square(q1-q2))) * 255)


def predictor(mode):
    def grade(engine, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), {"identity": True}
        a, b = thumb(target), thumb(reference)
        full = target.reshape((-1, 3)).astype(np.float64) / 255
        estimated, values = mode(a, b, full)
        assert np.isfinite(values).all()
        rgb = np.rint(255 * np.clip(values, 0, 1)).reshape(target.shape).astype(np.uint8)
        return finish_spatial(target, rgb, STRENGTH), {
            "distance_input": distance(a, b), "distance_mapped": distance(estimated.clip(0, 1), b),
            "clipped_fraction": float(np.any((values < 0) | (values > 1), axis=1).mean()),
        }
    return grade


def run(out):
    checked = verify(BASE, COUNTS)
    # The old scorers use this field to verify the frozen 13 archive, not to grade.
    class Identity:
        checkpoint_sha = CHECKPOINT
    engine = Identity()
    out.mkdir()
    results = {}
    sample = np.array([[[4, 20, 200], [255, 0, 128]]], dtype=np.uint8)
    for mode in (linear, project):
        process = predictor(mode)
        np.testing.assert_array_equal(process(engine, sample, sample.copy(), 255 - sample)[0], sample)
        root = out / mode.__name__
        root.mkdir()
        results[mode.__name__] = {
            k: evaluate(engine, root / k, process, mode.__name__)
            for k, evaluate in (("background", background), ("facd", facd), ("film_b", film_b), ("chart", chart))}
    json_to(out / "results.json", {"baseline_sha256": CHECKPOINT, "inputs": MANIFESTS,
             "preflight_counts": checked, "strength": STRENGTH, "arms": results})
    print({mode: {k: suite["passed"] for k, suite in suites.items()} for mode, suites in results.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
