"""Replay a measured tone-dependent edit contrast onto a different photo."""
import argparse
from pathlib import Path

import numpy as np

from audit_tst_reference_equivalence import HASHES, MANIFEST
from probe_tst_fixed_content_delta_supervision import FIXED, matched
from probe_tst_neutral_direction_fit import (
    BASE, LUMA, SPLIT, SPLIT_SHA, checked_rgb, finish_spatial, get_color, json_to, load, neutral, render, save,
)
from probe_tst_paired_donor_edit import donor_assets
from probe_tst_same_reference_style_pair import look_catalog, measure

ARMS = ("constant", "curve")
CENTERS = np.array([.1, .3, .5, .7, .9])


def midrank(luma):
    _, index, lengths = np.unique(luma, return_inverse=True, return_counts=True)
    centers = (lengths.cumsum() - .5 * lengths) / luma.size
    return centers[index].reshape(luma.shape)


def measured(reference):
    a, b = (img.astype(float) for img in reference)
    difference = .8 * (get_color(b) - get_color(a))
    rank = midrank((.5 * (a + b)) @ LUMA)
    assignment = np.minimum((rank * len(CENTERS)).astype(int), len(CENTERS) - 1)
    counts = [int((assignment == i).sum()) for i in range(len(CENTERS))]
    if min(counts) < 256:
        return None, counts
    return np.array([np.median(difference[assignment == i], axis=0) for i in range(len(CENTERS))]), counts


def replay(target, references, baseline):
    profile, counts = measured(references)
    if profile is None:
        return {name: [img.copy() for img in baseline] for name in ARMS}, {"counts": counts, "applied": False}
    mean = profile.mean(axis=0)
    rank = midrank(target @ LUMA)
    curve = np.stack([np.interp(rank, CENTERS, profile[:, index]) for index in range(2)], axis=2)
    curve += (mean - curve.mean(axis=(0, 1)))
    np.testing.assert_allclose(curve.mean(axis=(0, 1)), mean, atol=1e-10, rtol=0)
    midpoint = .5 * (get_color(baseline[0]) + get_color(baseline[1]))
    output, clipped = {}, {}
    for name, shift in zip(ARMS, (mean, curve), strict=True):
        pairs = [render(midpoint + sign * .5 * shift, image @ LUMA)
                 for sign, image in zip((-1, 1), baseline, strict=True)]
        output[name] = [frame for frame, _ in pairs]
        clipped[name] = [amount for _, amount in pairs]
    return output, {"applied": True, "counts": counts, "chroma_profile": profile.tolist(),
                    "spread_mean_chroma": mean.tolist(), "gamut_fraction": clipped}


def run(out):
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)["development"]
    catalog = look_catalog()
    entries = {row["original_path"]: row for row in load(MANIFEST, HASHES[2])["files"]}
    _, read = donor_assets()
    parent = BASE.parent / "tst-same-reference-style-pair-v1"
    frozen = load(parent / "results.json", FIXED)
    out.mkdir()
    tested = {}
    for key, row in frozen["cases"].items():
        part = split[int(key)]
        target = read(part["content"])[0]
        authored = [finish_spatial(target, read(part[tag])[0], .8) for tag in ("same_truth", "different_truth")]
        region = neutral(target)
        regions = {"whole": np.ones_like(region)}
        if region.sum() >= 256:
            regions["neutral"] = region
        for slot, value in row["slots"].items():
            if not value.get("arms", {}).get("13", {}).get("scored"):
                continue
            paths = matched(part, part["refs"][int(slot)], catalog, entries)
            references = [read(path)[0] for path in paths]
            earlier = value["arms"]["13"]
            ordinary = [checked_rgb(parent / f"{key}-{slot}-13-{i}.png", digest)
                        for i, digest in enumerate(earlier["hashes"])]
            candidates, audit = replay(target, references, ordinary)
            scores = {}
            for name, frames in candidates.items():
                metrics = {area: measure(frames, authored, mask) for area, mask in regions.items()}
                scoreable = [v["passed"] for section in metrics.values() for v in section.values() if v["scored"]]
                worst = max(float(np.abs(a.astype(float) - truth).mean() - np.abs(b.astype(float) - truth).mean())
                            for a, b, truth in zip(frames, ordinary, authored, strict=True))
                files = [save(out / f"{key}-{slot}-{name}-{i}.png", img) for i, img in enumerate(frames)]
                scores[name] = {"areas": metrics, "worst_added_mae": worst, "hashes": files,
                                "passed": bool(scoreable) and all(scoreable) and worst <= 2}
            tested.setdefault(key, {})[slot] = {"arms": scores, "audit": audit, "13": earlier}
    if len(tested) != 7 or sum(map(len, tested.values())) != 10:
        raise ValueError("Frozen cohort changed")
    summaries = {}
    for name in ARMS:
        groups = [[record["arms"][name] for record in slots.values()] for slots in tested.values()]
        summaries[name] = {"groups_pass": sum(all(slot["passed"] for slot in group) for group in groups),
                           "worst_added_mae": max(slot["worst_added_mae"] for group in groups for slot in group)}
    passed = (summaries["curve"]["groups_pass"] >= summaries["constant"]["groups_pass"] + 2
              and summaries["curve"]["worst_added_mae"] <= 2)
    json_to(out / "results.json", {"parent_sha256": FIXED, "summary": summaries, "cases": tested, "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
