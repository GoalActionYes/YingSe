"""Replay known reference-dependent neutral protection on the consumed TST split."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST, load
from probe_hdm_hdr_display import json_to
from probe_sparks_shared_chroma import LUMA, render
from probe_target_mirror_chroma import checked_rgb, panel, save
from probe_tst_authored_neutral_intent import RENDER_SHA, ROOT, get_color, proxy_masks
from spatial_render import finish_spatial
from train_tst_reference_equivalence import BASE, SPLIT, SPLIT_SHA

ASSAY = "beff9ef0b053f401cee583a2cfd09e62dda9f650d199b66cc777d66461f26618"
RESPONSE_SHA = "8339a64960c804cd5627f083cf5629ad0d54286482948bffc9068fbf11facdaf"
STATES = (0, 0, 1)


def masks(source, authored):
    _, _, switched = proxy_masks(source, authored)
    original = get_color(source)
    small = [np.abs(get_color(rgb) - original).max(axis=-1) <= 1 for rgb in authored]
    return [switched & state & ~other for state, other in (small, small[::-1])]


def measure(frames, authors, protection):
    a, b, c = map(get_color, frames)
    first, second = map(get_color, authors)
    first_keep, second_keep = protection
    selected = first_keep | second_keep
    expected = (second - first)[selected]
    actual = (c - .5 * (a + b))[selected]
    errors = {
        "preserve_mae": float(np.concatenate([(.5 * (np.abs(a - first) + np.abs(b - first)))[first_keep],
                                               np.abs(c - second)[second_keep]]).mean()),
        "edited_mae": float(np.concatenate([(.5 * (np.abs(a - first) + np.abs(b - first)))[second_keep],
                                             np.abs(c - second)[first_keep]]).mean()),
        "style_projection": float((actual * expected).sum() / np.square(expected).sum()),
        "style_delta_mae": float(np.abs(actual - expected).mean()),
        "same_style_swap_mae": float(np.abs((a - b)[selected]).mean()),
    }
    errors["passed"] = (errors["preserve_mae"] <= 1.5 and .75 <= errors["style_projection"] <= 1.25
                        and errors["style_delta_mae"] <= 3 and errors["same_style_swap_mae"] <= 1.5)
    return errors


def restore(source, images, masks):
    chroma = get_color(source)
    results = []
    for image, area in zip(images, masks, strict=True):
        result = image.copy()
        if area.any():
            recolored, _ = render(chroma[area], (image @ LUMA)[area])
            result[area] = recolored
        results.append(result)
    return results


def run(out):
    assay = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    cases = load(SPLIT / "private-assignment.json", SPLIT_SHA)["development"]
    entries = {item["original_path"]: item for item in load(MANIFEST, HASHES[2])["files"]}
    original_dir = BASE / "reference-equivalence-development-baseline-v1"
    original = load(original_dir / "results.json", RENDER_SHA)
    response_dir = BASE / "reference-equivalence-response-v1"
    response = load(response_dir / "results.json", RESPONSE_SHA)
    candidate = load(response_dir / "candidate/results.json", response["candidate_results_sha256"])
    if (assay["assignment_sha256"] != SPLIT_SHA or response["assignment_sha256"] != SPLIT_SHA
            or candidate["predictions"] == original["predictions"]):
        raise ValueError("Frozen cohort or render identity changed")

    def rgb(key):
        item = entries[key]
        pixels = checked_rgb(COHORT.parent / item["prepared_path"], item["sha256"])
        return np.asarray(Image.fromarray(pixels).resize((256, 256), Image.Resampling.BILINEAR))

    out.mkdir()
    report = {}
    for row in assay["cases"]:
        if not row["aligned"] or row["one_state_protects_pixels"] < 256:
            continue
        index = row["index"]
        entry = cases[index]
        target = rgb(entry["content"])
        authors = [finish_spatial(target, rgb(entry[side]), .8) for side in ("same_truth", "different_truth")]
        protections = masks(target, authors)
        selected = protections[0] | protections[1]
        if int(selected.sum()) != row["one_state_protects_pixels"]:
            raise ValueError("Frozen switch mask changed")
        paths = [f"{index}-{slot}.png" for slot in range(3)]
        plain = [checked_rgb(original_dir / name, original["predictions"][name]) for name in paths]
        tuned = [checked_rgb(response_dir / "candidate" / name, candidate["predictions"][name]) for name in paths]
        always = restore(target, plain, [selected] * len(STATES))
        conditional = restore(target, plain, [protections[side] for side in STATES])
        for before, after, side in zip(plain, conditional, STATES, strict=True):
            np.testing.assert_array_equal(before[protections[1 - side]], after[protections[1 - side]])
        arms = {"13": plain, "old_response": tuned, "always_protect": always, "context_protect": conditional}
        scores = {arm: measure(outputs, authors, protections) for arm, outputs in arms.items()}
        proof = {arm: [save(out / f"{index}-{arm}-{slot}.png", frame) for slot, frame in enumerate(arms[arm])]
                 for arm in ("always_protect", "context_protect")}
        comparison = panel(out / f"{index}-comparison.png", {
            "first look": {"target": target, "13": plain[0], "always": always[0],
                           "conditional": conditional[0], "author": authors[0]},
            "other look": {"target": target, "13": plain[2], "always": always[2],
                           "conditional": conditional[2], "author": authors[1]},
        })
        report[str(index)] = {"protect_counts": [int(area.sum()) for area in protections], "scores": scores,
                              "output_sha256": proof, "comparison_sha256": comparison}
    if len(report) != 5:
        raise ValueError("Changed five-case cohort")
    counts = {arm: sum(row["scores"][arm]["passed"] for row in report.values()) for arm in arms}
    json_to(out / "results.json", {"assay_sha256": ASSAY, "baseline_sha256": RENDER_SHA,
                                   "response_sha256": RESPONSE_SHA, "cases": report, "passing_cases": counts})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    run(cli.parse_args().out)
