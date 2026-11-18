"""Measure 13's response to the same official creative look on two real scenes."""
import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_sparks_archival_grade import ACES_DISPLAY, ACES_VIEW
from probe_tst_reference_spatial_signal import structure_score
from spatial_render import SpatialEngine, finish_spatial

CONFIG = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
LINEAR = "Linear ARRI Wide Gamut 3"
LOG = "ARRI LogC3 (EI800)"
LOOK_SHA = "cee155989194d1a63eb07454d894a400c9716248aa14b99e902cae8b66e33fa5"
SOURCES = {
    "showgirl_01": "25de638926f25c78f1d88dd643a3086e3447546fa05171eb0ea4b9b4dc420d3b",
    "poker_fullshot": "679b0aba8ef3b44e9edbde4683c4dde7d5dca0121f5d44deb6f5f25d50160393",
}
SPARKS_SHA = "9b7abd6a58a6dc5c7c05e0f246656182043c70de7560974d6099d7f59984e62d"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def converted(rgb, processor):
    value = rgb.copy(order="C")
    processor.applyRGB(value)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite color transform")
    return value


def endpoints(rgb):
    return float(np.mean((rgb < 0) | (rgb > 1)))


def quantize(rgb):
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def prepare(records, cube, output):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2
    import PyOpenColorIO as ocio

    cv2.setNumThreads(1)
    config = ocio.Config.CreateFromBuiltinConfig(CONFIG)
    transform = ocio.DisplayViewTransform(src=LOG, display=ACES_DISPLAY, view=ACES_VIEW)
    display = config.getProcessor(transform).getDefaultCPUProcessor()
    look = config.getProcessor(
        ocio.FileTransform(src=str(cube.resolve()), interpolation=ocio.INTERP_TETRAHEDRAL)
    ).getDefaultCPUProcessor()
    photos, audit = {}, {}
    for scene, (source, expected, color_space, height, width) in records.items():
        if digest(source) != expected:
            raise ValueError(f"Frozen source changed: {scene}")
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (height, width, 3) or image.dtype != np.float32:
            raise ValueError(f"Unexpected source geometry: {scene}")
        linear = cv2.resize(image[..., ::-1], (512, round(512 * height / width)), interpolation=cv2.INTER_AREA)
        encode = config.getProcessor(color_space, LOG).getDefaultCPUProcessor()
        direct = config.getProcessor(
            ocio.DisplayViewTransform(src=color_space, display=ACES_DISPLAY, view=ACES_VIEW)
        ).getDefaultCPUProcessor()
        log = converted(linear, encode)
        off = converted(log, display)
        on = converted(converted(log, look), display)
        direct_off = converted(linear, direct)
        if np.max(np.abs(quantize(off).astype(int) - quantize(direct_off))) > 1:
            raise ValueError("Neutral display path does not round trip")
        images = {"off": quantize(off), "on": quantize(on)}
        alignment = structure_score(images["off"], images["on"])["whole"]
        for role, rgb in images.items():
            Image.fromarray(rgb).save(output / f"{scene}-{role}.png")
        neutral = np.logical_and(*(target_regions(rgb, np.zeros(rgb.shape[:2], dtype=bool))["neutral_proxy"]
                                   for rgb in images.values()))
        change = float(np.abs(images["off"].astype(np.int16) - images["on"]).mean())
        photos[scene] = (images, neutral)
        audit[scene] = {
            "source_sha256": expected, "input_log_outside_unit_fraction": endpoints(log),
            "display_outside_unit_fraction": {"off": endpoints(off), "on": endpoints(on)},
            "shared_neutral_pixels": int(neutral.sum()), "known_edit_rgb_mae": change,
            "alignment": alignment,
            "images": {role: digest(output / f"{scene}-{role}.png") for role in images},
        }
    return photos, audit, {"ocio": ocio.__version__, "config": CONFIG, "config_id": config.getCacheID(),
                           "display": ACES_DISPLAY, "view": ACES_VIEW}


def metrics(original, expected, off, on, region):
    baseline, truth, off, on = (x[region].astype(np.float64) for x in (original, expected, off, on))
    signal, response = truth - baseline, on - off
    norm = np.sum(signal ** 2)
    product = float(np.sum(response * signal) / norm) if norm else None
    error = float(np.abs(response - signal).mean())
    return {"projection": product, "response_mae": error,
            "off_matches_original_mae": float(np.abs(off - baseline).mean()),
            "on_matches_authored_mae": float(np.abs(on - truth).mean()),
            "original_matches_authored_mae": float(np.abs(signal).mean()),
            "passed": product is not None and .75 <= product <= 1.25 and error <= 3}


def run(sources, cube, output, sparks_nam=None):
    if digest(cube) != LOOK_SHA:
        raise ValueError("Official look does not match the verified member")
    output.mkdir()
    records = {name: (sources / f"{name}-original.exr", expected, LINEAR, 1080, 1920)
               for name, expected in SOURCES.items()}
    if sparks_nam is not None:
        records.pop("showgirl_01")
        records["sparks_4592"] = (sparks_nam, SPARKS_SHA, "ACES2065-1", 2160, 4096)
    photos, audit, display = prepare(records, cube, output)
    result = {"official_lut_sha256": LOOK_SHA, "display": display, "prepared": audit, "predictions": {}}
    qualified = all(
        row["shared_neutral_pixels"] >= 256 and row["known_edit_rgb_mae"] >= 1
        and row["alignment"]["zero"] >= .65 and row["alignment"]["dx"] == row["alignment"]["dy"] == 0
        and (sparks_nam is None or row["input_log_outside_unit_fraction"] <= .01)
        for row in audit.values()
    )
    result["qualified"] = qualified
    if qualified:
        import torch
        torch.set_num_threads(4)
        engine = SpatialEngine("multiref-v1")
        result["weight_sha256"] = engine.checkpoint_sha
        for target, reference in (tuple(records), tuple(reversed(records))):
            images, neutral = photos[target]
            original = images["off"]
            expected = finish_spatial(original, images["on"], .8)
            predictions, hashes = {}, {}
            for role, style in photos[reference][0].items():
                prediction = finish_spatial(original, engine.prepare(original, style).apply(original), .8)
                path = output / f"{target}-13-reference-{role}.png"
                Image.fromarray(prediction).save(path)
                predictions[role], hashes[role] = prediction, digest(path)
            result["predictions"][target] = {
                "reference": reference, "hashes": hashes,
                "whole": metrics(original, expected, *predictions.values(), np.ones(neutral.shape, dtype=bool)),
                "shared_neutral": metrics(original, expected, *predictions.values(), neutral),
            }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("cube", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sparks-nam", type=Path)
    args = parser.parse_args()
    run(args.sources, args.cube, args.output, args.sparks_nam)
