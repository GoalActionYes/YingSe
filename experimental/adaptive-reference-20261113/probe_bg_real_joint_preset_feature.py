"""Evaluate frozen, joint two-image preset features without the external renderer."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_bg_real_crossed_grade_training import (
    BASE, EDITS, ROOT, describe, digest, feature, fit, grade, headroom, mask_for,
    pair, panel, read, scores, target_regions,
)
from probe_bg_real_paired_contrast import write_outputs
from probe_crossed_edit_pair_supervision import fixed_sets
from probe_dast_exposure_response import save_report
from probe_deep_preset_two_image import load_model

BASELINE_SHA = "cbd75c0361b4a52068e9527dfb5a886c919843daf320434113dba2e6951de47b"
ARMS = ("global", "joint")


def input_tensor(rgb):
    image = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BICUBIC)
    pixels = np.asarray(image).astype(np.float32) / 127.5 - 1
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).cuda()


@torch.inference_mode()
def features(model, target, target_summary, reference):
    plain = feature(target_summary, describe(reference))
    activations = torch.cat((target, input_tensor(reference)), dim=1)
    for layer in range(1, model.depth + 1):
        activations = getattr(model, f"tconv_{layer}")(activations)
    if activations.shape[1] != 256 or not torch.isfinite(activations).all():
        raise ValueError("Unexpected encoder features")
    stats = torch.cat((activations.mean(dim=(2, 3)), activations.std(dim=(2, 3), unbiased=False)), dim=1)
    return {"global": plain, "joint": np.concatenate((plain, stats[0].double().cpu().numpy()))}


def training(model, source):
    observations = {arm: [] for arm in ARMS}
    labels = []
    for index, group in enumerate(source):
        image = source[(index + 1) % len(source)]["C1"]
        target, summary = input_tensor(image), describe(image)
        for edit, reference in product(EDITS.values(), group.values()):
            measured = features(model, target, summary, grade(reference, edit))
            for arm, row in measured.items():
                observations[arm].append(row)
            labels.append(edit)
    if len(labels) != 96:
        raise ValueError("Unexpected training budget")
    return {arm: fit(rows, labels) for arm, rows in observations.items()}


def run(assets, output):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_sha = load_model(assets)
    fit_data, held, registered = fixed_sets()
    evidence = ROOT / "docs/research/2026-11-19-bg-real-crossed-grade-training-evidence.json"
    if digest(evidence) != BASELINE_SHA:
        raise ValueError("Expected frozen global results")
    prior = json.loads(evidence.read_bytes())["cases"]
    fitted = training(model, fit_data)
    output.mkdir()
    cases = {}
    for example in held.values():
        key = example["target_pair"]
        entry = registered[key]
        original = read(entry["input_file"])
        canvas = headroom(original)
        target, summary = input_tensor(canvas), describe(canvas)
        refs = pair(BASE / "bg-real-pixel-inputs", example["input"])
        prior_panel = BASE / "bg-real-crossed-grade-training" / f"{key}.png"
        if digest(prior_panel) != prior[key]["panel_sha256"]:
            raise ValueError("Global panel changed")
        with Image.open(prior_panel) as image:
            control = np.asarray(image.convert("RGB"))
        renders = {arm: {role: {} for role in refs} for arm in ARMS}
        estimates = {arm: {role: {} for role in refs} for arm in ARMS}
        for row, (role, photo) in enumerate(refs.items()):
            for col, (style, operation) in enumerate(EDITS.items()):
                review = features(model, target, summary, grade(photo, operation))
                for arm, value in review.items():
                    center, scale, bias, weights = fitted[arm]
                    decision = np.clip((value - center) / scale @ weights + bias, [-1, 0], [1, 1])
                    estimate = grade(canvas, decision, .8)
                    renders[arm][role][style] = estimate
                    estimates[arm][role][style] = decision.tolist()
                    if arm == "global":
                        x, y = 8 + col * 264, 26 + (3 + row) * 280
                        np.testing.assert_array_equal(estimate, control[y:y + 256, x:x + 256])

        human = mask_for(entry, key.split("-")[0])
        neutral = target_regions(original, human)["neutral_proxy"] & human
        truth = {style: grade(canvas, coeff, .8) for style, coeff in EDITS.items()}
        dest = output / key
        dest.mkdir()
        hashes = {arm: write_outputs(dest / arm, data) for arm, data in renders.items()}
        board = {"prescribed edit": truth}
        board.update({f"{arm} / {role}": photos for arm, data in renders.items() for role, photos in data.items()})
        panel_path = dest / "comparison.png"
        panel(panel_path, board)
        values = {arm: scores(data, truth, human, neutral) for arm, data in renders.items()}
        worst = max(values["joint"]["styles"][style]["response"][role]["target_mae"]
                    - values["global"]["styles"][style]["response"][role]["target_mae"]
                    for role, style in product(refs, EDITS))
        cases[key] = {**values, "estimated": estimates, "worst_regression": worst,
                      "output_sha256": hashes, "panel_sha256": digest(panel_path)}
    totals = {arm: sum(row[arm]["passed"] for row in cases.values()) for arm in ARMS}
    worst = max(row["worst_regression"] for row in cases.values())
    save_report(output / "results.json", {"encoder_asset_sha256": model_sha, "cases": cases, "totals": totals,
                "feature_count": {arm: len(model[0]) for arm, model in fitted.items()},
                "worst_regression": worst, "passed": totals["joint"] >= 3 and worst <= 2.})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("assets", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
