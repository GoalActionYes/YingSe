"""Calibrate rendered chroma strength from ordinary two-image color quantiles."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_beyond_rgb_matched_cast import compare
from probe_multilum_gray_probe import chroma, valid
from probe_physical_blue_global_fit import display, linear
from probe_target_luma_edit_mix import BEYOND, LUMA, PARENT, beyond_inputs, mid_inputs
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb,
    digest, finish_spatial, json_to, panel, save,
)


LABELS = ("base", "real", "edit")


def calibrate(target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"gain": 0., "reason": "identity"}
    selected = [valid(image) for image in (target, reference, baseline)]
    common = selected[0] & selected[2]
    if min(common.sum(), selected[1].sum()) < 256:
        return baseline.copy(), {"gain": 1., "reason": "support"}
    colors = [chroma(image) for image in (target, reference, baseline)]
    axis = np.median((colors[2] - colors[0])[common], axis=0)
    norm = np.linalg.norm(axis)
    if norm < .03:
        return baseline.copy(), {"gain": 1., "reason": "axis"}
    a, b, c = [np.quantile(plane[mask] @ (axis / norm), [.25, .50, .75])
               for plane, mask in zip(colors, selected, strict=True)]
    change = c - a
    magnitude = float(change @ change)
    if magnitude < 3 * .03 ** 2:
        return baseline.copy(), {"gain": 1., "reason": "response"}
    gain = float(np.clip(((b - a) @ change) / magnitude, 0, 2))
    cb = colors[0] + gain * (colors[2] - colors[0])
    scales = np.exp(np.stack((cb[..., 0], np.zeros_like(cb[..., 0]), cb[..., 1]), axis=-1))
    y = linear(baseline) @ LUMA
    adjusted = scales * (y / (scales @ LUMA))[..., None]
    peak = adjusted.max(axis=-1)
    damp = np.divide(1 - y, peak - y, out=np.ones_like(y), where=peak > 1)
    output = y[..., None] + (adjusted - y[..., None]) * damp[..., None]
    error = float(np.max(np.abs(output @ LUMA - y)))
    if error > 1e-12:
        raise ValueError("Luma invariant failed")
    return display(output), {"gain": gain, "reason": "calibrated", "max_linear_luma_change": error}


def archive(cohort, name):
    if cohort is mid_inputs:
        root = BASE / "mid-matched-wall-cast-v1"
        report = checked_json(root / "results.json", PARENT)
        rows = report["cases"][name]["models"]["13"]["hashes"]
        names = dict(zip(LABELS, ("base", "captured", "edited"), strict=True))
        guide_files = {key: (f"reference-{state}.png", report["inputs"][state]) for key, state in names.items()}
        output_files = {key: (f"{name}-13-{state}.png", rows[state]) for key, state in names.items()}
    else:
        root = BASE / "beyond-rgb-matched-cast-first-v1"
        report = checked_json(root / "report.json", BEYOND)
        row = report["cases"][name]
        donor = row["reference"]
        guide_files = {key: (f"{donor}-guide-{key}.png", report["references"][donor]["sha256"][key]) for key in LABELS}
        output_files = {key: (f"{name}-spatial13-{key}.png", row["outputs_sha256"][f"{name}-spatial13-{key}.png"])
                        for key in LABELS}
    return [{key: checked_rgb(root / file, sha) for key, (file, sha) in table.items()}
            for table in (guide_files, output_files)]


def display_chroma(rgb):
    return rgb - (rgb @ LUMA)[..., None]


def run(out):
    out.mkdir()
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed archived checkpoint")
    cases = {}
    for cohort in (mid_inputs, beyond_inputs):
        for name, group, target, expected, regions, _ in cohort():
            guides, original = archive(cohort, name)
            updated, audit = {}, {}
            for key in LABELS:
                updated[key], audit[key] = calibrate(target, guides[key], original[key])
            same, _ = calibrate(target, target, original["base"])
            if not np.array_equal(same, target):
                raise ValueError("Identity invariant failed")
            exposure = display(linear(target) * .65)
            old_exposure = finish_spatial(target, engine.prepare(target, exposure).apply(target), STRENGTH)
            new_exposure, audit["exposure"] = calibrate(target, exposure, old_exposure)
            target_chroma = display_chroma(target)
            exposure_mae = [float(np.abs(display_chroma(frame) - target_chroma).mean())
                            for frame in (old_exposure, new_exposure)]
            scores = {}
            for zone, mask in regions.items():
                scores[zone] = {}
                for model, states in (("13", original), ("calibrated", updated)):
                    row = compare(target, expected, *states.values(), mask)
                    row["passed"] = row["both_pass"] and row["style"]["off_matches_original_mae"] <= 3
                    scores[zone][model] = row
            output = updated | {"exposure-13": old_exposure, "exposure-calibrated": new_exposure}
            cases[name] = dict(group=group, audit=audit, zones=scores, exposure_chroma_mae=exposure_mae,
                               hashes={key: save(out / f"{name}-{key}.png", frame) for key, frame in output.items()},
                               panel_sha=panel(out / f"{name}-comparison.png",
                                               {"reference": guides, "target": {"source": target, "desired edit": expected},
                                                "13": original, "calibrated": updated, "exposure": output}))
    src = Path(__file__)
    json_to(out / "results.json", dict(parents={"mid": PARENT, "beyond": BEYOND}, spatial=CHECKPOINT,
            script_sha=digest(src), protocol_sha=digest(src.with_name("two-image-chroma-quantile-gain-protocol.md")),
            cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
