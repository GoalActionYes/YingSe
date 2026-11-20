"""Evaluate whether renderer disagreement carries transferable color-edit information."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_beyond_rgb_matched_cast import compare
from probe_physical_blue_global_fit import display, linear
from probe_target_luma_edit_mix import BEYOND, LUMA, PARENT, beyond_inputs, mid_inputs
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save
from probe_two_image_chroma_quantile_gain import LABELS, archive
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import analyze


def stable_archive(cohort, name):
    if cohort is mid_inputs:
        root = BASE / "mid-matched-wall-cast-v1"
        index = checked_json(root / "results.json", PARENT)
        hashes = index["cases"][name]["models"]["stable"]["hashes"]
        files = {key: (f"{name}-stable-{tag}.png", hashes[tag]) for key, tag in
                 zip(LABELS, ("base", "captured", "edited"), strict=True)}
    else:
        root = BASE / "beyond-rgb-matched-cast-first-v1"
        index = checked_json(root / "report.json", BEYOND)
        hashes = index["cases"][name]["outputs_sha256"]
        files = {key: (f"{name}-stable-{key}.png", hashes[f"{name}-stable-{key}.png"]) for key in LABELS}
    return {key: checked_rgb(root / file, sha) for key, (file, sha) in files.items()}


def features(target, spatial, stable):
    source = target.astype(np.float64)
    return np.concatenate((spatial - source, stable - source), axis=-1).reshape(-1, 6)


def solve(gram, cross):
    results = {}
    for label, columns in (("one", 3), ("two", 6)):
        matrix = gram[:columns, :columns]
        penalty = .01 * np.trace(matrix) / columns
        results[label] = np.linalg.solve(matrix + penalty * np.eye(columns), cross[:columns])
    return results


def learn(cohort):
    gram, cross = np.zeros((6, 6)), np.zeros((6, 3))
    for name, _, target, expected, _, _ in cohort():
        _, spatial = archive(cohort, name)
        stable = stable_archive(cohort, name)
        for key in LABELS:
            x = features(target, spatial[key], stable[key])
            gram += (x.T @ x) / len(x)
            if key == "edit":
                y = (expected.astype(np.float64) - target).reshape(-1, 3)
                cross += (x.T @ y) / len(x)
    return solve(gram, cross)


def predict(target, spatial, stable, weights):
    columns = len(weights)
    change = (features(target, spatial, stable)[:, :columns] @ weights).reshape(target.shape)
    return np.rint(np.clip(target + change, 0, 255)).astype(np.uint8)


def run(out, control_sha):
    folds = {cohort: learn(other) for cohort, other in ((mid_inputs, beyond_inputs), (beyond_inputs, mid_inputs))}
    root = BASE / "two-image-chroma-quantile-gain-v1"
    control = checked_json(root / "results.json", control_sha)
    out.mkdir()
    cases = {}
    for cohort, weights in folds.items():
        for name, group, target, expected, regions, _ in cohort():
            guides, spatial = archive(cohort, name)
            stable = stable_archive(cohort, name)
            stacks = {"13": spatial}
            for label, weights_for_arm in weights.items():
                stacks[label] = {key: predict(target, spatial[key], stable[key], weights_for_arm) for key in LABELS}
                if not np.array_equal(predict(target, target, target, weights_for_arm), target):
                    raise ValueError("Same-image feature regression is not identity")
            target_path = out / f"{name}-exposure-target.png"
            reference_path = out / f"{name}-exposure-guide.png"
            exposure = display(linear(target) * .65)
            save(target_path, target)
            save(reference_path, exposure)
            control_file = f"{name}-exposure-13.png"
            exp13 = checked_rgb(root / control_file, control["cases"][name]["hashes"]["exposure-13"])
            exp_stable = apply_mapping(target, analyze(target_path, reference_path), .8)
            exp_arms = {key: predict(target, exp13, exp_stable, w) for key, w in weights.items()}
            tonal = {key: float(np.abs((image.astype(float) - exp13) @ LUMA).mean()) for key, image in exp_arms.items()}
            scores = {zone: {key: compare(target, expected, *images.values(), mask)
                             for key, images in stacks.items()} for zone, mask in regions.items()}
            for row in scores.values():
                for score in row.values():
                    score["passed"] = score["both_pass"] and score["style"]["off_matches_original_mae"] <= 3
            frames = {f"{key}-{state}": image for key in weights for state, image in stacks[key].items()}
            frames.update({f"{key}-exposure": image for key, image in exp_arms.items()})
            cases[name] = dict(group=group, exposure_luma_error=tonal, scores=scores,
                hashes={label: save(out / f"{name}-{label}.png", image) for label, image in frames.items()},
                panel_sha=panel(out / f"{name}-comparison.png",
                                {"reference": guides, "target": {"original": target, "desired": expected}, **stacks}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parents={"mid": PARENT, "beyond": BEYOND, "control": control_sha},
        fitted={cohort.__name__: {key: w.tolist() for key, w in weights.items()} for cohort, weights in folds.items()},
        script_sha=digest(script), protocol_sha=digest(script.with_name("cross-renderer-capture-regression-protocol.md")),
        cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("control_sha")
    run(**vars(parser.parse_args()))
