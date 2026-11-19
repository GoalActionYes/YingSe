"""Test a two-image chroma branch using a reference with skin-weighted color."""

import argparse
from functools import partial
from pathlib import Path
import re
import sys

import numpy as np
import torch

from audit_bg_real_skin_regions import PROTOCOL
from probe_bg_real_legacy_skin import protector
from probe_frozen_candidate_development_crosscheck import COUNTS, film_b
from probe_sparks_shared_chroma import LUMA, chroma, render
from probe_stable_chroma_hybrid import chart
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, LUMA_INTEGER, ROOT, STRENGTH, background, checked_json, checked_rgb, digest, facd, json_to,
)
from spatial_render import SpatialEngine, finish_spatial
from verify_development_inputs import MANIFESTS, verify

LABEL = "Reference skin chroma"
REGIONS_SHA = "2e0fe1e6b0b47e3579263bbd7cee626713955fee6c3cde71ef8323784de702b1"
BG_SHA = "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d"


def weight_for(model, rgb):
    low, _, stats = model._predict(rgb)
    if not stats["skin_detected"]:
        return None
    full = model._cv.resize(low, (rgb.shape[1], rgb.shape[0]), interpolation=model._cv.INTER_LINEAR)
    if not np.isfinite(full).all() or full.min() < -1e-6 or full.max() > 1 + 1e-6:
        raise ValueError("Invalid skin weighting")
    return full.clip(0, 1)


def grade(engine, target, reference, baseline, *, model):
    if np.array_equal(target, reference):
        return target.copy(), {"route": "same_image"}
    target_weight = weight_for(model, target)
    if target_weight is None:
        return baseline.copy(), {"route": "no_target_skin"}
    guide_weight = weight_for(model, reference)
    if guide_weight is None:
        return baseline.copy(), {"route": "no_reference_skin"}
    guide_y = reference @ LUMA
    gray_guide_color = chroma(reference, guide_y) * guide_weight[..., None]
    grade_rgb, direct, limited = render_branch(engine, target, baseline, target_weight, gray_guide_color, guide_y)
    return grade_rgb, {"route": "applied", "target_weight_above_half": float((target_weight > .5).mean()),
                       "reference_weight_above_half": float((guide_weight > .5).mean()),
                       "max_luma_error": direct, "gamut_limited_fraction": limited,
                       "changed_pixels": int(np.any(grade_rgb != baseline, axis=-1).sum())}


def render_branch(engine, target, baseline, target_weight, guide_color, guide_y):
    guide, _ = render(guide_color, guide_y)
    alternate = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
    old_y = baseline @ LUMA
    old_color = chroma(baseline, old_y)
    new_color = chroma(alternate, alternate @ LUMA)
    grade_rgb, limited = render(old_color + target_weight[..., None] * (new_color - old_color), old_y)
    grade_rgb[target_weight == 0] = baseline[target_weight == 0]
    direct = np.abs((grade_rgb.astype(np.int32) - baseline) @ LUMA_INTEGER) / 10000
    if direct.max() > .5:
        raise ValueError("Luminance changed beyond RGB8 quantization")
    return grade_rgb, float(direct.max()), limited


def skin_boxes(out, suite):
    if digest(PROTOCOL) != REGIONS_SHA:
        raise ValueError("Existing measurement regions changed")
    boxes = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                       PROTOCOL.read_text(encoding="utf-8"), re.MULTILINE)
    folder = BASE / "bg-real-editor-look"
    receipt = checked_json(folder / "results.json", BG_SHA)
    rows = {}
    for left, right, label, x, y, x1, y1 in boxes:
        pair = f"{left}-{right}"
        area = np.s_[int(y):int(y1), int(x):int(x1)]
        measures = {}
        for state in ("off", "on"):
            variants = {}
            for arm in ("baseline", "candidate"):
                pictures = []
                for role in ("C1", "F1"):
                    tag = f"{role}-{state}"
                    if arm == "baseline":
                        filename = f"{tag}-13.png"
                        frame = checked_rgb(folder / pair / filename, receipt["cases"][pair]["artifacts"][filename])
                    else:
                        frame = checked_rgb(out / "background" / pair / f"{tag}.png",
                                            suite["cases"][pair]["outputs"][tag]["sha256"])
                    pictures.append(frame[area])
                delta = pictures[1].astype(float) - pictures[0]
                variants[arm] = float(np.abs(chroma(delta, delta @ LUMA)).mean())
            measures[state] = variants
        rows.setdefault(pair, {})[label] = measures
    if len(boxes) != 9:
        raise ValueError("Lost a frozen skin or material region")
    return rows


@torch.inference_mode()
def run(out):
    checked = verify(BASE, COUNTS)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Baseline checkpoint changed")
    path = ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py"
    model = protector(path)
    candidate = partial(grade, model=model)
    sample = np.full((64, 64, 3), 128, np.uint8)
    np.testing.assert_array_equal(candidate(engine, sample, sample.copy(), 255 - sample)[0], sample)
    untouched, detail = candidate(engine, sample, 255 - sample, 255 - sample)
    np.testing.assert_array_equal(untouched, 255 - sample)
    if detail["route"] != "no_target_skin":
        raise ValueError("Flat negative did not use the no-skin fallback")
    out.mkdir()
    suites = {name: suite(engine, out / name, candidate, LABEL)
              for name, suite in (("background", background), ("facd", facd), ("film_b", film_b), ("chart", chart))}
    json_to(out / "results.json", {"baseline_sha256": CHECKPOINT, "input_manifests": MANIFESTS,
                                  "preflight_counts": checked, "skin_source_sha256": digest(path),
                                  "suites": suites, "registered_regions": skin_boxes(out, suites["background"]),
                                  "passed": all(data["passed"] for data in suites.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("locator_dependencies", type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    run(args.out)
