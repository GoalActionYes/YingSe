"""Offline replay of held-out luminance while preserving predicted linear RGB ratios."""

import argparse
import json
from pathlib import Path

import numpy as np

from probe_physical_blue_global_fit import CAPTURE_SHA, display, inside, linear, pixels, save, score, verified
from probe_physical_blue_spatial_cross import within


PARENT_SHA = "e6b593e4267ec311abcbf0076567c3874fb1ede5144f7d0f166d2ed5859a157c"
LUMA = np.array([.2126, .7152, .0722])


def align(prediction, truth):
    original = linear(prediction)
    predicted_y = original @ LUMA
    wanted_y = linear(truth) @ LUMA
    # Black has no defined channel ratio. Preserve those pixels and account for them.
    found = predicted_y > 0
    scale = np.divide(wanted_y, predicted_y, out=np.ones_like(predicted_y), where=found)
    matched = original * scale[..., None]
    np.testing.assert_allclose((matched @ LUMA)[found], wanted_y[found], atol=1e-12, rtol=0)
    return display(matched), found, (matched > 1).any(axis=-1)


def run(base, out):
    archive = base / "physical-blue-spatial-cross-v1"
    capture = base / "chromamill-capture-cross-v1"
    frozen = json.loads(verified(archive / "results.json", PARENT_SHA).read_bytes())
    captures = json.loads(verified(capture / "results.json", CAPTURE_SHA).read_bytes())
    out.mkdir()
    cases = {}
    for tag, row in frozen["cases"].items():
        scene, _, _, level = tag.split("-")
        white, blue = (pixels(capture / f"{scene}-{light}-{level}-off.png",
                              captures["prepared"][scene][f"{light}-{level}"]["hashes"]["off"])
                       for light in ("white", "blue"))
        digital = display(linear(white) * np.array(row["gains"]))
        result = {}
        for name, truth, gates in (("physical", blue, (3, 8)), ("digital", digital, (1, 3))):
            valid = inside(white) & inside(truth)
            result[name] = {}
            for arm, identity in row["controls"][name]["arms"].items():
                filename = f"{tag}-{name}-{arm}.png"
                prediction = pixels(archive / filename, identity["sha256"])
                initial = score(prediction, truth, valid)
                if initial != identity["scores"]:
                    raise ValueError("Archived score does not match the recorded baseline")
                rendered, found, clipped = align(prediction, truth)
                usable = valid & found & ~clipped
                measured = score(rendered, truth, valid)
                result[name][arm] = {
                    "original": initial, "aligned": measured,
                    "whole": score(rendered, truth, np.ones(valid.shape, bool)),
                    "eligible_unclipped": score(rendered, truth, usable) if usable.any() else None,
                    "eligible_clipped_fraction": float(clipped[valid].mean()),
                    "eligible_undefined_fraction": float((~found)[valid].mean()),
                    "eligible_luminance_mae_linear": float(np.abs((linear(rendered) - linear(truth)) @ LUMA)[valid].mean()),
                    "passed": within(measured, *gates), "sha256": save(out / filename, rendered),
                }
        cases[tag] = result
    summary = {name: {arm: sum(item[name][arm]["passed"] for item in cases.values())
                      for arm in ("global", "spatial")} for name in ("physical", "digital")}
    report = {"parent_sha256": PARENT_SHA, "cases": cases, "pass_counts": summary}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
