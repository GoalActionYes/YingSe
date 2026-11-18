"""Run color-ratio preserving brightness controls through the normal two-image inference."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_dast_exposure_response import STATES, midtone, statistics, verified_png
from spatial_render import SpatialEngine, finish_spatial

INPUTS_SHA = "7e5137f4e559daf39e4f4f09a7500637beb897713257c4d7101de20a0d9dbf97"
BASELINE_SHA = "892a26347772b7649adb73593cc1740b5efac9bdafc51aff548aa5b4d32aec0c"
Y = np.array([.2126, .7152, .0722])
ANALYSIS_SIZE = (256, 256)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def linear(rgb):
    x = rgb / 255.
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def srgb(values):
    x = values.clip(0, 1)
    return 255 * np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)


def radiometric_match(base, requested):
    rank = np.argsort((base @ Y).ravel(), kind="stable")
    wanted = np.empty(len(rank), dtype=np.float64)
    wanted[rank] = np.sort((requested @ Y).ravel())
    wanted = wanted.reshape(base.shape[:2])
    source = linear(base)
    maximum = source.max(axis=-1)
    high = np.divide(1, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    low = np.zeros_like(high)
    headroom = srgb(source * high[..., None]) @ Y
    clipped = wanted > headroom + 1e-8
    for _ in range(24):
        mid = (high + low) * .5
        under = srgb(source * mid[..., None]) @ Y < wanted
        low = np.where(under, mid, low)
        high = np.where(under, high, mid)
    control = np.rint(srgb(source * ((low + high) * .5)[..., None])).astype(np.uint8)
    distribution_error = float(np.abs(np.sort((control @ Y).ravel()) - np.sort(wanted.ravel())).mean())
    return control, distribution_error, float(clipped.mean())


def save(path, pixels):
    Image.fromarray(pixels).save(path)
    return digest(path)


def resized(src, expected, binary=False):
    pixels = verified_png(src, expected)
    mode = Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR
    return np.array(Image.fromarray(pixels).resize(ANALYSIS_SIZE, mode))


def project(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    dot = b @ b
    return float(a @ b / dot) if dot else None


@torch.inference_mode()
def run(prep, baseline, out):
    prepared_plan, base_results = prep / "inputs.json", baseline / "results.json"
    if (digest(prepared_plan), digest(base_results)) != (INPUTS_SHA, BASELINE_SHA):
        raise ValueError("Exposure parent changed")
    data = json.loads(prepared_plan.read_bytes())["groups"]
    saved = json.loads(base_results.read_bytes())
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != saved["weight_sha256"]:
        raise ValueError("Unexpected 13 model")
    out.mkdir()
    groups = []
    for row, prior in zip(data, saved["groups"], strict=True):
        group = row["group"]
        d = prep / str(group)
        folder = out / str(group)
        folder.mkdir()
        ref = resized(d / "nominal.png", row["frames"]["nominal"])
        ref_roi = resized(d / "nominal-mask.png", row["target_mask_sha256"], binary=True) > 0
        identity, _, _ = radiometric_match(ref, ref)
        np.testing.assert_array_equal(identity, ref)
        target = data[row["target_group"] - 1]
        target_folder = prep / str(target["group"])
        rgb = verified_png(target_folder / "nominal.png", target["frames"]["nominal"])
        target_roi = verified_png(target_folder / "nominal-mask.png", target["target_mask_sha256"]) > 0
        anchor = verified_png(baseline / str(group) / "nominal.png", prior["output_sha256"]["nominal"])
        nominal_replay = finish_spatial(rgb, engine.prepare(rgb, ref).apply(rgb), .8)
        np.testing.assert_array_equal(nominal_replay, anchor)
        checks = {}
        for state in STATES[2:]:
            actual_input = resized(d / f"{state}.png", row["frames"][state])
            control, y_mae, clipped = radiometric_match(ref, actual_input)
            mask = ref_roi & midtone(ref) & midtone(control)
            inputs = statistics(ref, control, mask)
            result = finish_spatial(rgb, engine.prepare(rgb, control).apply(rgb), .8)
            actual = verified_png(baseline / str(group) / f"{state}.png", prior["output_sha256"][state])
            region = target_roi & midtone(rgb) & midtone(anchor) & midtone(actual) & midtone(result)
            target_control = statistics(anchor, result, region)
            target_actual = statistics(anchor, actual, region)
            valid_input = y_mae <= 2 and inputs["qualified"] and inputs["chroma_norm"] <= .015
            score = None
            if target_control["qualified"] and target_actual["qualified"]:
                score = project(target_control["mean_log_chroma_change"], target_actual["mean_log_chroma_change"])
            checks[state] = {
                "matching_y_mae": y_mae, "luma_not_attainable_fraction": clipped, "source": inputs,
                "target_actual": target_actual, "target_control": target_control, "relative_projection": score,
                "eligible": bool(valid_input and score is not None),
                "reproduced": bool(valid_input and score is not None and score >= .5
                                   and target_control["chroma_norm"] >= .03),
                "reference_sha256": save(folder / f"{state}-reference.png", control),
                "output_sha256": save(folder / f"{state}-output.png", result),
            }
        groups.append({"group": group, "exposures": checks})
    values = [row for group in groups for row in group["exposures"].values()]
    eligible, reproduced = (sum(val[key] for val in values) for key in ("eligible", "reproduced"))
    report = {
        "input_sha256": INPUTS_SHA, "baseline_sha256": BASELINE_SHA, "model_sha256": engine.checkpoint_sha,
        "evaluated": len(values), "eligible": eligible, "reproduced": reproduced,
        "supports_luma_only_propagation": eligible >= 12 and reproduced / eligible >= .75, "groups": groups,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "baseline", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.prepared, args.baseline, args.output)
