"""Replay fixed cubic maps across photos using verified reference before/after pairs."""

import argparse
from pathlib import Path

import numpy as np

from probe_aligned_cubic_pilot import cubic_mapping, learn, thumbnail_pair
from probe_arri_film_b_dose import endpoints
from probe_arri_shared_look import metrics
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_frozen_candidate_development_crosscheck import BASE, COUNTS, STRENGTH
from probe_frozen_render_blend_bound import FULL, FULL_SHA, neutral
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb, panel, save
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial
from verify_development_inputs import manifest, verify


def paired_fit(original, edited):
    if original.shape != edited.shape:
        raise ValueError("Verified reference pair has different sizes")
    a, b = thumbnail_pair(original, edited)
    yy, xx = np.indices(a.shape[:2])
    train = (xx // 8 + yy // 8) % 2 == 0
    weights = learn(a, b, train.ravel())
    if weights is None:
        raise ValueError("Unstable fit on paired reference")
    table = cubic_mapping(weights)
    error = np.abs(apply_mapping(a, table, 1).astype(np.int16) - b)
    blocks = [error[y:y + 8, x:x + 8].mean() for y in range(0, a.shape[0], 8) for x in range(0, a.shape[1], 8)]
    mae, p95, tile = float(error[~train].mean()), float(np.quantile(error[~train], .95)), float(max(blocks))
    return table, {"mae": mae, "p95": p95, "max_tile": tile, "within_original_limit": mae <= 1.5 and p95 <= 4 and tile <= 6}


def apply(target, table):
    return finish_spatial(target, apply_mapping(target, table, 1), STRENGTH)


def background(out, original):
    out.mkdir()
    parents = verified_json(PARENT / "results.json", PARENT_SHA)["cases"]
    entries = {row["pair"]: row for row in parents}
    cases = {}
    for key, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        directory = BASE / "bg-real-editor-look" / key
        assets = row["artifacts"]
        target, expected = (checked_rgb(directory / name, assets[name]) for name in ("target.png", "expected-on.png"))
        images, fits, hashes, board = {}, {}, {}, {}
        for variant in ("C1", "F1"):
            filenames = (f"{variant}-off-reference.png", f"{variant}-on-reference.png")
            off, on = (checked_rgb(directory / name, assets[name]) for name in filenames)
            table, fits[variant] = paired_fit(off, on)
            result = apply(target, table)
            images[variant] = {"off": target, "on": result}
            hashes[variant] = save(out / f"{key}-{variant}.png", result)
            old_output = original[key]["outputs"][f"{variant}-on"]["sha256"]
            baseline = checked_rgb(BASE / FULL / "background" / key / f"{variant}-on.png", old_output)
            board[variant] = {"target": target, "reference": on, "expanded": baseline,
                              "paired fit": result, "known edit": expected}
        person = mask_for(entries[key], key.split("-")[0])
        candidate = evaluate(target, {"off": target, "on": expected}, images, person)
        protected = person & neutral(target)
        absolute_error = {variant: float(np.abs(images[variant]["on"][protected].astype(np.int16) - expected[protected]).mean())
                          for variant in images} if protected.sum() >= 256 else None
        cases[key] = {"reference_fits": fits, "candidate": candidate, "candidate_neutral_person_mae": absolute_error,
                      "expanded": original[key]["candidate"], "outputs_sha256": hashes,
                      "comparison_sha256": panel(out / f"{key}-comparison.png", board)}
    count = sum(row["candidate"]["passed"] for row in cases.values())
    worst = max(row["candidate"]["max_protected_swap"] for row in cases.values())
    return {"cases": cases, "passing_cases": count, "max_protected_swap": worst, "passed": count >= 3 and worst <= 3}


def film(out, original):
    out.mkdir()
    prep = manifest(BASE, "arri-two-looks-v2")
    sources = {name: endpoints(BASE / "arri-two-looks-v2", prep, name) for name in prep["sources"]}
    fitted = {name: paired_fit(*images) for name, images in sources.items()}
    cases = {}
    for key, row in manifest(BASE, "arri-film-b-crossed-v1")["cases"].items():
        scene, donor = row["target"], row["reference"]
        target, style = sources[scene]
        table, _ = fitted[donor]
        result = apply(target, table)
        expected = finish_spatial(target, style, STRENGTH)
        region = neutral(target) & neutral(style)
        if region.sum() < 256:
            raise ValueError("Missing Film B common neutral support")
        scores = {label: metrics(target, expected, target, result, area)
                  for label, area in (("whole", np.ones(region.shape, bool)), ("neutral", region))}
        prefix = f"{scene}-from-{donor}"
        baseline = checked_rgb(BASE / FULL / "film_b" / f"{prefix}-film_b.png", original[key]["outputs"]["film_b"]["sha256"])
        cases[key] = {"candidate": scores, "expanded": original[key]["candidate"], "passed": all(s["passed"] for s in scores.values()),
                      "output_sha256": save(out / f"{prefix}.png", result),
                      "comparison_sha256": panel(out / f"{prefix}-comparison.png", {
                          "paired reference": {"target": target, "reference": sources[donor][1], "expanded": baseline,
                                               "paired fit": result, "known edit": expected}})}
    count = sum(row["passed"] for row in cases.values())
    return {"reference_fits": {name: pair[1] for name, pair in fitted.items()},
            "cases": cases, "passing_cases": count, "passed": count == 6}


def run(out):
    verify(BASE, COUNTS)
    frozen = verified_json(BASE / FULL / "results.json", FULL_SHA)
    if frozen["strength"] != STRENGTH:
        raise ValueError("Frozen render strength changed")
    out.mkdir()
    results = {"background": background(out / "background", frozen["suites"]["background"]["cases"]),
               "film_b": film(out / "film_b", frozen["suites"]["film_b"]["cases"])}
    json_to(out / "results.json", {"archive_sha256": FULL_SHA, "strength": STRENGTH, "suites": results,
                                   "passed": all(suite["passed"] for suite in results.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
