"""Screen a common highlight chromaticity estimate on consumed development cases."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_gray_edge_correction import LUMA, PARENT_SHA, authored, controlled
from probe_multilum_chromaticity import linear_rgb
from probe_reference_chroma_spread import BASE, PARENT, decision


def extract_lines(linear):
    points, directions = [], []
    for row in range(0, linear.shape[0] - 15, 16):
        for col in range(0, linear.shape[1] - 15, 16):
            pixels = linear[row:row + 16, col:col + 16].reshape(256, 3)
            intensity = pixels.sum(axis=1)
            valid = (intensity >= .06) & (pixels.max(axis=1) < .97)
            if valid.sum() < 224:
                continue
            ordered = np.argsort(intensity[valid], kind='stable')
            intensity = intensity[valid][ordered]
            chroma = pixels[valid][ordered, :2] / intensity[:, None]
            quarter = len(intensity) // 4
            dim, bright = chroma[:quarter].mean(axis=0), chroma[-quarter:].mean(axis=0)
            axis = bright - dim
            length = np.linalg.norm(axis)
            if intensity[-quarter:].mean() - intensity[:quarter].mean() < .12 or length < .025:
                continue
            axis /= length
            normal = np.array([-axis[1], axis[0]])
            if np.quantile(np.abs((chroma - bright) @ normal), .8) <= .012:
                points.append(bright)
                directions.append(axis)
    return np.asarray(points), np.asarray(directions)


def highlight_chroma(linear):
    points, axes = extract_lines(linear)
    count = len(points)
    if count < 4:
        return None, count, 0
    normals = np.column_stack((-axes[:, 1], axes[:, 0]))
    offsets = np.sum(normals * points, axis=1)
    first, second = np.triu_indices(count, k=1)
    a, b = normals[first], normals[second]
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    select = np.abs(determinant) >= .25
    a, b, determinant = a[select], b[select], determinant[select]
    c, d = offsets[first[select]], offsets[second[select]]
    candidates = np.column_stack((c * b[:, 1] - a[:, 1] * d, a[:, 0] * d - c * b[:, 0])) / determinant[:, None]
    candidates = candidates[(candidates.min(axis=1) >= .05) & (candidates.sum(axis=1) <= .95)]
    if not len(candidates):
        return None, count, 0

    def support(values):
        residual = np.abs(values @ normals.T - offsets)
        forwards = values @ axes.T - np.sum(points * axes, axis=1)
        return (residual <= .015) & (forwards >= -.010), residual

    support_mask, residual = support(candidates)
    totals = support_mask.sum(axis=1)
    losses = (residual * support_mask).sum(axis=1)
    best = np.lexsort((losses, -totals))[0]
    minimum = max(4, (count + 2) // 3)
    if totals[best] < minimum:
        return None, count, int(totals[best])
    chosen = support_mask[best]
    system = normals[chosen]
    point = np.linalg.lstsq(system, offsets[chosen], rcond=None)[0]
    accepted = support(point[None])[0][0] & chosen
    used = int(accepted.sum())
    if (used < minimum or point.min() < .05 or point.sum() > .95
            or np.linalg.eigvalsh(normals[accepted].T @ normals[accepted] / used)[0] < .10):
        return None, count, used
    return np.r_[point, 1 - point.sum()], count, used


def estimate(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    return highlight_chroma(linear_rgb(np.asarray(image)))


def correction(target, reference, baseline):
    roles = dict(target=target, reference=reference, baseline=baseline)
    estimates = {name: estimate(rgb) for name, rgb in roles.items()}
    debug = dict(estimates={name: dict(chroma=None if color is None else color.tolist(),
                                     lines=lines, support=support)
                            for name, (color, lines, support) in estimates.items()},
                 applied=False, clipped_channel_fraction=0.)
    if np.array_equal(target, reference):
        return target.copy(), debug
    if any(result[0] is None for result in estimates.values()):
        return baseline.copy(), debug
    logs = [np.log(estimates[name][0]) for name in roles]
    source, look, current = [value - value.mean() for value in logs]
    gain = np.clip(source + .8 * (look - source) - current, -.15, .15)
    raw = linear_rgb(baseline)
    adjusted = raw * np.exp(gain)
    luma = adjusted @ LUMA
    adjusted *= np.divide(raw @ LUMA, luma, out=np.ones_like(luma), where=luma > 0)[..., None]
    debug.update(applied=True, gain=gain.tolist(), clipped_channel_fraction=float(np.mean(adjusted > 1)))
    adjusted = adjusted.clip(0, 1)
    encoded = np.where(adjusted <= .0031308, 12.92 * adjusted, 1.055 * adjusted ** (1 / 2.4) - .055)
    if not np.isfinite(encoded).all():
        raise ValueError('Nonfinite output')
    return np.rint(encoded * 255).astype(np.uint8), debug


def check_synthetic():
    light = np.array([.39, .34, .27])
    materials = np.array([[.76, .16, .08], [.09, .78, .13], [.07, .11, .82], [.12, .35, .53]])
    ramp = np.linspace(0, 1, 256).reshape(16, 16, 1)
    patches = [.25 * material + ramp * light for material in materials]
    estimate_color, lines, support = highlight_chroma(np.concatenate(patches, axis=1))
    np.testing.assert_allclose(estimate_color, light, atol=1e-10, rtol=0)
    diffuse = [(ramp + .25) * material for material in materials]
    assert highlight_chroma(np.concatenate(diffuse, axis=1))[0] is None
    return dict(estimated=estimate_color.tolist(), expected=light.tolist(), lines=lines, support=support)


def run(out):
    synthetic = check_synthetic()
    manifest = PARENT / 'results.json'
    if digest(manifest) != PARENT_SHA:
        raise ValueError('Background intervention baseline changed')
    parent = json.loads(manifest.read_text(encoding='utf-8'))
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    cases = controlled(out, parent, correction)
    evidence = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-psbench-action-pair-evidence.json'
    comparisons = authored(out, BASE / 'psbench-shared-local-v1', evidence, correction)
    elapsed = time.monotonic() - started
    if elapsed >= 180:
        raise TimeoutError('Frozen screening budget exceeded')
    result = dict(controlled=decision(cases), cases=cases, authored=comparisons, synthetic=synthetic,
                  elapsed_seconds=elapsed, parent_sha256=PARENT_SHA, model_sha256=parent['model_sha256'],
                  pair_sha256=digest(evidence), script_sha256=digest(__file__),
                  runner_sha256=digest(Path(__file__).with_name('probe_gray_edge_correction.py')),
                  protocol_sha256=digest(Path(__file__).with_name('highlight-consensus-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(controlled=result['controlled'], seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
