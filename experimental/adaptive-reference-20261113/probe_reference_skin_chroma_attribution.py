"""Separate reference mask and luma effects on the frozen chroma-only branch."""

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import torch

from probe_reference_skin_chroma_isolation import (
    BASE, BG_SHA, CHECKPOINT, LUMA, PROTOCOL, REGIONS_SHA, ROOT, SpatialEngine, checked_json,
    checked_rgb, chroma, digest, grade, json_to, protector, render_branch, weight_for,
)
from probe_target_mirror_chroma import save

OLD_SHA = "a2977240820be621b6c4acfddb1cb37b2798269cbead4b4bd8c07b5f48b2fe3f"
FACTORS = ("mask", "luma", "both")
CORE = {"785-3553": {"面部小块"}, "9483-19432": {"前额", "手背"}}


def regions():
    if digest(PROTOCOL) != REGIONS_SHA:
        raise ValueError("Frozen measurement regions changed")
    boxes = {}
    for left, right, name, x, y, x1, y1 in re.findall(
        r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
        PROTOCOL.read_text(encoding="utf-8"), re.MULTILINE,
    ):
        boxes.setdefault(f"{left}-{right}", {})[name] = np.s_[int(y):int(y1), int(x):int(x1)]
    if sum(len(row) for row in boxes.values()) != 9:
        raise ValueError("Nine registered regions expected")
    return boxes


def score(anchor, frame, area):
    delta = frame[area].astype(float) - anchor[area]
    return float(np.abs(chroma(delta, delta @ LUMA)).mean())


def enough(before, after):
    return before - after >= .25 and after <= .75 * before


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    anchor = BASE / "bg-real-editor-look"
    previous = BASE / "reference-skin-chroma-isolation-v1"
    frozen = checked_json(anchor / "results.json", BG_SHA)
    old = checked_json(previous / "results.json", OLD_SHA)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or old["baseline_sha256"] != CHECKPOINT:
        raise ValueError("Changed 13 weights")
    model = protector(ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py")
    boxes = regions()
    out.mkdir()
    cases = {}
    primary = {arm: 0 for arm in FACTORS}
    observations = 0
    for pair, item in frozen["cases"].items():
        folder = out / pair
        folder.mkdir()
        source = anchor / pair
        receipts = item["artifacts"]
        target = checked_rgb(source / "target.png", receipts["target.png"])
        target_weight = weight_for(model, target)
        cases[pair] = {}
        for state in ("off", "on"):
            refs, baselines, original, guide_weights = [], [], [], []
            for background in ("C1", "F1"):
                name = f"{background}-{state}"
                ref_name, base_name = f"{name}-reference.png", f"{name}-13.png"
                ref = checked_rgb(source / ref_name, receipts[ref_name])
                baseline = checked_rgb(source / base_name, receipts[base_name])
                actual, debug = grade(engine, target, ref, baseline, model=model)
                stored = old["suites"]["background"]["cases"][pair]["outputs"][name]
                saved = checked_rgb(previous / "background" / pair / f"{name}.png", stored["sha256"])
                np.testing.assert_array_equal(actual, saved)
                if debug["route"] != "applied" or stored["route"] != "applied" or target_weight is None:
                    raise ValueError("The frozen branch no longer activates")
                refs.append(ref)
                baselines.append(baseline)
                original.append(saved)
                guide_weights.append(weight_for(model, ref))
            if refs[0].shape != refs[1].shape or any(weight is None for weight in guide_weights):
                raise ValueError("The two guides no longer share their original canvas")
            y = [ref @ LUMA for ref in refs]
            f_color = chroma(refs[1], y[1])
            choices = {"mask": (guide_weights[0], y[1]), "luma": (guide_weights[1], y[0]),
                       "both": (guide_weights[0], y[0])}
            attempts = {}
            hashes = {}
            for name, (weight, light) in choices.items():
                frame, _, _ = render_branch(engine, target, baselines[1], target_weight, f_color * weight[..., None], light)
                attempts[name] = frame
                hashes[name] = save(folder / f"{state}-{name}.png", frame)
            metrics = {}
            for name, area in boxes[pair].items():
                before = score(*original, area)
                older = old["registered_regions"][pair][name][state]["candidate"]
                if before != older:
                    raise ValueError("Old region score changed")
                trials = {arm: score(original[0], frame, area) for arm, frame in attempts.items()}
                metrics[name] = {"frozen": before, **trials}
                if name in CORE.get(pair, ()):
                    observations += 1
                    for arm, value in trials.items():
                        primary[arm] += int(enough(before, value))
            cases[pair][state] = {"regions": metrics, "hashes": hashes,
                                  "ref_skin_weight_mae": float(np.abs(guide_weights[1] - guide_weights[0]).mean()),
                                  "ref_luma_mae": float(np.abs(y[1] - y[0]).mean())}
    if observations != 6:
        raise ValueError("Frozen core must contain six measurements")
    fit = [arm for arm in ("mask", "luma") if primary[arm] >= 5]
    json_to(out / "results.json", {"parent_sha256": OLD_SHA, "baseline_sha256": CHECKPOINT,
                                  "observations": observations, "primary_counts": primary,
                                  "leading_factor": fit[0] if len(fit) == 1 else None, "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("locator_dependencies", type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    run(args.out)
