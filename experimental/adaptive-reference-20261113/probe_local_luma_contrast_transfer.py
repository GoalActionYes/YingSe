"""Test whether reference middle-scale luminance detail restores intended contrast."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_hdm_hdr_display import json_to
from probe_target_mirror_chroma import BG_SHA, FACD_SHA, background, facd
from probe_target_mirror_chroma import CHECKPOINT, LUMA, STRENGTH, chroma, render
from spatial_render import SpatialEngine


def box_axis(values, radius, axis):
    pad = [(0, 0)] * values.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(values, pad, mode="edge")
    integral = np.cumsum(padded, axis=axis, dtype=np.float64)
    pad[axis] = (1, 0)
    integral = np.pad(integral, pad)
    window = 2 * radius + 1
    leading = [slice(None)] * values.ndim
    trailing = leading.copy()
    leading[axis] = slice(window, None)
    trailing[axis] = slice(None, -window)
    return (integral[tuple(leading)] - integral[tuple(trailing)]) / window


def detail(luma, first, second):
    narrow = box_axis(box_axis(luma, first, 0), first, 1)
    broad = box_axis(box_axis(luma, second, 0), second, 1)
    return narrow - broad


def amplitude(rgb):
    sample = Image.fromarray(rgb)
    sample.thumbnail((256, 256), Image.Resampling.BILINEAR)
    band = detail(np.asarray(sample) @ LUMA, 4, 16)
    return float(np.quantile(np.abs(band), .90))


def correction(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True}
    origin_scale, reference_scale, model_scale = map(amplitude, (target, reference, baseline))
    gain = float(np.clip((reference_scale - model_scale) / max(origin_scale, 4.), -.5, .5))
    y = baseline @ LUMA
    factor = max(target.shape[:2]) / 256
    band = detail(target @ LUMA, max(1, round(4 * factor)), max(1, round(16 * factor)))
    requested = y + gain * band
    result, compressed = render(chroma(baseline, y), requested.clip(0, 255))
    return result, {
        "target_band_rgb8": origin_scale, "reference_band_rgb8": reference_scale,
        "baseline_band_rgb8": model_scale, "detail_gain": gain,
        "output_change_mae": float(np.abs(result.astype(np.int16) - baseline).mean()),
        "luma_change_mae": float(np.abs(result @ LUMA - y).mean()), "gamut_fraction": compressed,
        "luma_clip_fraction": float(((requested < 0) | (requested > 255)).mean()),
    }


def verify():
    value = np.arange(21, dtype=np.float64).reshape(3, 7)
    for radius in (1, 4):
        for axis in (0, 1):
            transposed = np.moveaxis(value, axis, 0)
            reflected = np.pad(transposed, ((radius, radius), (0, 0)), mode="edge")
            expected = np.stack([reflected[i:i + 2 * radius + 1].mean(axis=0)
                                 for i in range(len(transposed))], axis=0)
            np.testing.assert_allclose(box_axis(value, radius, axis), np.moveaxis(expected, 0, axis), atol=1e-12)
    flat = np.full((13, 21, 3), 150, dtype=np.uint8)
    same, _ = correction(None, flat, flat.copy(), 255 - flat)
    np.testing.assert_array_equal(same, flat)
    unaltered, _ = correction(None, flat, flat + 10, 255 - flat)
    np.testing.assert_array_equal(unaltered, 255 - flat)


def run(output):
    verify()
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13 checkpoint")
    output.mkdir()
    bg, sdr = (probe(engine, output / folder, correction, "contrast band")
               for probe, folder in ((background, "bg-real"), (facd, "facd")))
    directions = {}
    for case, details in bg["cases"].items():
        rows = {}
        for arm in ("C1", "F1"):
            old = details["baseline"]["projection"][arm]["luma"]
            new = details["candidate"]["projection"][arm]["luma"]
            rows[arm] = {"baseline": old, "candidate": new,
                         "improvement_toward_expected": abs(1 - old) - abs(1 - new)}
        directions[case] = rows
    improvements = [arm["improvement_toward_expected"] for case in directions.values() for arm in case.values()]
    count = sum(x >= .10 for x in improvements)
    direction_gate = count >= 6 and min(improvements) >= -.05
    json_to(output / "results.json", {
        "checkpoint_sha256": CHECKPOINT, "strength": STRENGTH,
        "background_baseline_sha256": BG_SHA, "facd_baseline_sha256": FACD_SHA,
        "background": bg, "facd": sdr, "luma_directions": directions, "improvements_at_least_point_one": count,
        "direction_gate": direction_gate, "passed": direction_gate and bg["passed"] and sdr["passed"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
