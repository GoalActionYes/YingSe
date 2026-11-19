"""Test frozen separate shadow/highlight reference histograms against reversed assignment."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_bg_real_style_cross import ROLES
from probe_tst_neutral_direction_fit import checked_rgb, get_color, json_to, load, render, save
from probe_sparks_shared_chroma import LUMA
from probe_two_tone_edges import AXIS, BASE, PARENT, PARENT_SHA, ROOT, STYLES, mask_for, read, target_regions, versions
from region_histogram import weighted_histogram
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from training_checks import finite_output

FROZEN = "6f96270d626be1c7f820988b15784c1e71ec653e039d83c1b40342b2417361a9"
SOURCE_SHA = "83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9"
ARMS = ("matched", "reversed")


def shade_pair(engine, target, reference, baseline, support_weights=None):
    if np.array_equal(target, reference):
        return {name: target.copy() for name in ARMS}, {"identity": True}
    target_light = target @ LUMA
    shadow, highlight = np.quantile(target_light, [.25, .75])
    content, guide = engine.analysis(target), engine.analysis(reference)
    histogram, convert = (engine.api[name] for name in ("batch_histogram_1D", "srgb_to_ycbcr"))
    yc, ys = convert(content), convert(guide)
    lower = ys[:, :1] <= ys[:, :1].median()
    mass = float(lower.float().mean())
    if min(mass, 1 - mass) < .10 or highlight - shadow < 8:
        return {name: baseline.copy() for name in ARMS}, {"applied": False, "dark_fraction": mass}
    selected = (lower, ~lower)
    metadata = {"dark_fraction": mass}
    histogram_values = ys
    if support_weights is not None:
        selected, detail = support_weights(lower)
        metadata.update(detail)
        if selected is None:
            return {name: baseline.copy() for name in ARMS}, {"applied": False, **metadata}
        histogram_values = ys.cpu()
    low = engine.api["TF"].resize(content, (64, 64), antialias=True)
    target_hist = histogram(yc)
    full_reference = histogram(ys)
    framed = []
    for selector in selected:
        weights = selector.to(device=histogram_values.device, dtype=torch.float32)
        subset = weighted_histogram(histogram_values, weights).to(ys.device)
        reference_hist = torch.cat((full_reference[:, :1], subset[:, 1:]), dim=1)
        grid = engine.api["lowpass"](engine.model.apply_grid(low, torch.cat((target_hist, reference_hist), dim=1)))
        finite_output(grid, (1, 13, 8, 16, 16))
        frame = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8)
        framed.append(get_color(frame))
    weight = np.clip((target_light - shadow) / (highlight - shadow), 0, 1)[..., None]
    origin = get_color(baseline)
    luma = baseline @ LUMA
    a, b = framed
    outputs, fractions = {}, {}
    for name, (low_color, high_color) in zip(ARMS, ((a, b), (b, a)), strict=True):
        change = (1 - weight) * low_color + weight * high_color - origin
        change -= change.mean(axis=(0, 1))
        outputs[name], fractions[name] = render(origin + change, luma)
    return outputs, {"applied": True, **metadata, "gamut_fractions": fractions}


def switch_error(frames, truth, area):
    change = (frames["light-cool"].astype(float) - frames["dark-cool"]) @ AXIS
    want = (truth["light-cool"].astype(float) - truth["dark-cool"]) @ AXIS
    error = np.abs(change - want)
    return {"whole": float(error.mean()), "neutral": float(error[area].mean()) if area.sum() >= 256 else None}


@torch.inference_mode()
def run(out, predict=shade_pair):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    source = load(ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json", SOURCE_SHA)
    old_folder = BASE / "two-tone-edge"
    previous = load(old_folder / "results.json", FROZEN)
    cohort = {row["pair"]: row for row in load(PARENT / "results.json", PARENT_SHA)["cases"]}
    if engine.checkpoint_sha != source["checkpoint_sha256"]:
        raise ValueError("Baseline weights changed")
    flat = np.full((32, 32, 3), 128, np.uint8)
    for output in predict(engine, flat, flat.copy(), 255 - flat)[0].values():
        np.testing.assert_array_equal(output, flat)
    flat_ref = 255 - flat
    for output in predict(engine, flat, flat_ref, flat_ref)[0].values():
        np.testing.assert_array_equal(output, flat_ref)
    out.mkdir()
    cases = {}
    for item in source["cases"].values():
        key = item["target_pair"]
        row = cohort[key]
        target_styles = versions(read(row["input_file"]))
        target = target_styles["base"]
        truth = {name: finish_spatial(target, rgb, .8) for name, rgb in target_styles.items()}
        person = mask_for(row, key.split("-")[0])
        area = target_regions(target, person)["neutral_proxy"] & person
        summaries, checks = {}, {}
        for role in ROLES:
            asset = item["input"][role]
            refs = versions(checked_rgb(BASE / "bg-real-pixel-inputs" / asset["name"], asset["sha256"]))
            frames = {name: {} for name in ("13", *ARMS)}
            metadata, hashes = {}, {}
            for style in STYLES:
                filename = f"{key}-{role}-{style}-13.png"
                expected_sha = previous["cases"][key]["roles"][role]["outputs"][style]["13"]
                baseline = checked_rgb(old_folder / filename, expected_sha)
                control = finish_spatial(target, engine.prepare(target, refs[style]).apply(target), .8)
                np.testing.assert_array_equal(control, baseline)
                frames["13"][style] = baseline
                changed, metadata[style] = predict(engine, target, refs[style], baseline)
                hashes[style] = {}
                for name, img in changed.items():
                    frames[name][style] = img
                    hashes[style][name] = save(out / f"{key}-{role}-{style}-{name}.png", img)
            stats = {name: switch_error(pictures, truth, area) for name, pictures in frames.items()}
            base = stats["13"]
            if base != previous["cases"][key]["roles"][role]["13"]:
                raise ValueError("Frozen switch score changed")
            for name in ARMS:
                trial = stats[name]
                trial["passed"] = (trial["neutral"] is not None
                                   and base["neutral"] - trial["neutral"] >= max(.5, .2 * base["neutral"])
                                   and trial["whole"] <= base["whole"] + .5)
            checks[role] = {name: {style: (shots[style] @ AXIS)[area] for style in ("uniform", "base")}
                            for name, shots in frames.items()}
            summaries[role] = {"scores": stats, "hashes": hashes, "metadata": metadata}
        control = {}
        if area.sum() >= 256:
            delta = ((truth["uniform"].astype(float) - truth["base"]) @ AXIS)[area]
            for name in ("13", *ARMS):
                control[name] = {
                    "swap": float(np.abs(checks["C1"][name]["base"] - checks["F1"][name]["base"]).mean()),
                    "uniform": {role: float(np.abs(x[name]["uniform"] - x[name]["base"] - delta).mean())
                                for role, x in checks.items()},
                }
            before = control["13"]
            for name in ARMS:
                after = control[name]
                after["passed"] = after["swap"] <= before["swap"] + 1 and all(
                    after["uniform"][role] <= before["uniform"][role] + 1 for role in ROLES
                )
        cases[key] = {"roles": summaries, "neutral_controls": control}
    counts = {name: sum(role["scores"][name]["passed"] for case in cases.values() for role in case["roles"].values())
              for name in ARMS}
    passes = {name: all(case["neutral_controls"].get(name, {}).get("passed", False) for case in cases.values())
              for name in ARMS}
    json_to(out / "results.json", {"cases": cases, "switch_pass": counts, "control_pass": passes,
                                  "parent_sha256": FROZEN,
                                  "passed": counts["matched"] == 8 and counts["reversed"] <= 6 and passes["matched"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
