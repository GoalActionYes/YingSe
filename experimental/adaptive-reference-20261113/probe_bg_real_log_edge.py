"""Test a local linear-light ratio cue against edits and real capture exposures."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_bg_real_crossed_grade_training import BASE, EDITS, grade, pair
from probe_contrast_code_gap import export
from probe_crossed_edit_pair_supervision import fixed_sets
from probe_dast_exposure_response import STATES, save_report, verified_png
from probe_dast_local_edit_cross import INPUT_SHA, read_ledger

REC709 = np.array([.2126, .7152, .0722])
VARIANTS = {"lossless": None, "jpeg95": 95}


def measure_linear(rgb):
    light = rgb @ REC709
    total = rgb.sum(axis=2, keepdims=True)
    color = np.divide(rgb, total, where=total > 0, out=np.zeros_like(rgb))
    midrange = (light >= .035) & (light <= .80)
    log_light = np.log(np.maximum(light, 1e-12))
    edges = []
    for axis in (0, 1):
        levels = np.moveaxis(log_light, axis, 0)
        qualified = np.moveaxis(midrange, axis, 0)
        hue = np.moveaxis(color, axis, 0)
        like = np.max(np.abs(hue[:-1] - hue[1:]), axis=2) <= .035
        mask = qualified[:-1] & qualified[1:] & like
        edges.append(np.abs(levels[:-1] - levels[1:])[mask])
    delta = np.concatenate(edges)
    return {"pairs": int(delta.size), "score": float(np.quantile(delta, .75)) if delta.size >= 1024 else None}


def scores(rgb):
    observed = {}
    for name, quality in VARIANTS.items():
        published = Image.fromarray(export(rgb, quality, False))
        published.thumbnail((256, 256), Image.Resampling.BILINEAR)
        code = np.asarray(published, dtype=float) / 255
        linear = np.where(code <= .04045, code / 12.92, ((code + .055) / 1.055) ** 2.4)
        observed[name] = measure_linear(linear)
    return observed


def collect(folders):
    return {(role, style): scores(grade(rgb, operation)) for role, rgb in folders.items()
            for style, operation in EDITS.items()}


def would_flag(row, threshold):
    return row["score"] is not None and row["score"] > threshold


def compare_pairwise(cases, name):
    comparisons = []
    for key, case in cases.items():
        rows = {(r["role"], r["edit"]): r["scores"][name]["score"] for r in case}
        if any(value is None for value in rows.values()):
            comparisons.append({"case": key, "qualified": False})
            continue
        contrast = {role: rows[role, "contrast"] - rows[role, "none"] for role in ("C1", "F1")}
        swap = max(abs(rows["C1", edit] - rows["F1", edit]) for edit in EDITS)
        comparisons.append({"case": key, "qualified": True, "contrast_change": contrast,
                            "largest_background_change": swap, "both_stronger_than_background": min(contrast.values()) > swap})
    return comparisons


def run(output):
    y, x = np.indices((64, 64))
    neutral = np.broadcast_to((.10 + .04 * ((x + y) % 2))[..., None], (64, 64, 3))
    control = measure_linear(neutral)
    brightened = measure_linear(neutral * 1.6)
    np.testing.assert_allclose(control["score"], brightened["score"], atol=1e-12, rtol=0)
    if control["pairs"] != brightened["pairs"]:
        raise ValueError("Linear exposure sanity check failed")
    training, held, _ = fixed_sets()
    negatives = [v["score"] for photo in training for (_, edit), row in collect(photo).items() if edit != "contrast"
                 for v in row.values() if v["score"] is not None]
    threshold = max(negatives)
    cases = {item["target_pair"]: [{"role": role, "edit": edit, "scores": value}
                                  for (role, edit), value in collect(pair(BASE / "bg-real-pixel-inputs", item["input"])).items()]
             for item in held.values()}
    prepared = BASE / "dast-exposure-prepared-v1"
    captures = []
    for row in read_ledger(prepared, "inputs.json", INPUT_SHA)["groups"]:
        for state in STATES:
            frame = verified_png(prepared / str(row["group"]) / f"{state}.png", row["frames"][state])
            captures.append({"group": row["group"], "exposure": state, "scores": scores(frame)})
    summary = {}
    for name in VARIANTS:
        rows = [row for case in cases.values() for row in case]
        matches = sum(row["edit"] == "contrast" and would_flag(row["scores"][name], threshold) for row in rows)
        others = sum(row["edit"] != "contrast" and would_flag(row["scores"][name], threshold) for row in rows)
        false_capture = sum(would_flag(row["scores"][name], threshold) for row in captures)
        summary[name] = {"contrast_detected": matches, "other_edit_false_positives": others,
                         "capture_false_positives": false_capture, "paired_background": compare_pairwise(cases, name),
                         "passed": matches >= 6 and others <= 2 and false_capture <= 3}
    output.mkdir()
    save_report(output / "results.json", {"threshold": threshold, "training_negative_scores": len(negatives),
                "synthetic_exposure_control": [control, brightened], "results": summary, "cases": cases,
                "captures": captures, "passed": all(item["passed"] for item in summary.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
