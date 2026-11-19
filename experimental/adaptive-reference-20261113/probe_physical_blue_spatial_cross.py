"""Test whether a position-dependent color transform transfers across light levels."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from probe_physical_blue_global_fit import (
    CAPTURE_SHA, DIGITAL_SHA, SCENES, display, inside, linear, pixels, save, score, verified,
)


def features(rgb, spatial):
    image = linear(rgb)
    if not spatial:
        return image
    height, width = rgb.shape[:2]
    x = np.linspace(-1, 1, width)[None, :, None]
    y = np.linspace(-1, 1, height)[:, None, None]
    return np.concatenate((image, image * x, image * y), axis=-1)


def predict(source, desired, target, spatial):
    fit_mask = inside(source) & inside(desired)
    design = features(source, spatial)[fit_mask]
    matrix, _, rank, _ = np.linalg.lstsq(design, linear(desired)[fit_mask], rcond=None)
    if rank != design.shape[1] or len(design) < 512:
        raise ValueError("Invalid fit support or rank")
    rendered = display(features(target, spatial) @ matrix)
    return rendered, {"training_pixels": len(design), "matrix": matrix.tolist()}


def within(metrics, mae, p95):
    return metrics["rgb8_mae"] <= mae and metrics["pixel_mean_channel_error_p95"] <= p95


def run(base, out):
    capture = base / "chromamill-capture-cross-v1"
    digital = base / "cti-selective-correction-v1"
    captures = json.loads(verified(capture / "results.json", CAPTURE_SHA).read_bytes())
    edits = json.loads(verified(digital / "results.json", DIGITAL_SHA).read_bytes())
    out.mkdir()
    cases = {}
    for scene, level in product(SCENES, (213, 241)):
        held = 454 - level
        donor = next(name for name in SCENES if name != scene)
        choice = edits["digital"][donor][str(level)]
        if not choice["qualified"]:
            raise ValueError("Unqualified digital control")
        gains = np.array(choice["gains"])
        frames = {light: [pixels(capture / f"{scene}-{light}-{strength}-off.png",
                                 captures["prepared"][scene][f"{light}-{strength}"]["hashes"]["off"])
                          for strength in (level, held)] for light in ("white", "blue")}
        source, target = frames["white"]
        if any(image.shape != source.shape for pair in frames.values() for image in pair):
            raise ValueError("Frozen images differ in size")
        reference_digital = pixels(digital / f"{donor}-{level}-off-reference.png",
                                   choice["states"]["off"]["hashes"]["reference"])
        digital_source, digital_test = [display(linear(image) * gains) for image in (source, target)]
        np.testing.assert_array_equal(digital_source, reference_digital)
        tag = f"{scene}-{level}-to-{held}"
        result = {"gains": gains.tolist(), "controls": {}}
        for name, (desired, truth) in (("physical", frames["blue"]),
                                       ("digital", (digital_source, digital_test))):
            valid = inside(target) & inside(truth)
            if valid.sum() < 512:
                raise ValueError("Insufficient holdout support")
            arms = {}
            for arm, spatial in (("global", False), ("spatial", True)):
                rendered, audit = predict(source, desired, target, spatial)
                arms[arm] = audit | {"scores": score(rendered, truth, valid),
                                     "whole": score(rendered, truth, np.ones(valid.shape, dtype=bool)),
                                     "sha256": save(out / f"{tag}-{name}-{arm}.png", rendered)}
            result["controls"][name] = {"arms": arms, "eligible_fraction": float(valid.mean())}
        real, design = (result["controls"][name]["arms"] for name in ("physical", "digital"))
        real_global, real_spatial = (real[arm]["scores"] for arm in ("global", "spatial"))
        digital_global, digital_spatial = (design[arm]["scores"] for arm in ("global", "spatial"))
        improvement = 1 - real_spatial["rgb8_mae"] / real_global["rgb8_mae"]
        result.update(physical_mae_reduction=improvement,
                      passed=within(real_spatial, 3, 8) and improvement >= .15
                      and within(digital_global, 1, 3) and within(digital_spatial, 1, 3)
                      and digital_spatial["rgb8_mae"] - digital_global["rgb8_mae"] <= .25)
        cases[tag] = result
    report = {"capture_sha256": CAPTURE_SHA, "digital_sha256": DIGITAL_SHA, "cases": cases,
              "passed": all(row["passed"] for row in cases.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
