"""Replay a known donor edit with the target author's own fit as a control."""
import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from audit_tst_reference_equivalence import COHORT, HASHES, INDEX, MANIFEST, family
from probe_aligned_cubic_pilot import cubic_mapping, learn
from probe_tst_neutral_switch_fit import BASE, RENDER_SHA, SPLIT, SPLIT_SHA, neutral, structure_score
from probe_tst_neutral_switch_renders import checked_rgb, finish_spatial, json_to, load, save
from reference_color_app.lut import apply_mapping


def edit_number(name):
    return PurePosixPath(name).stem.rsplit("_", 1)[1]


def replay(original, edited, target, train):
    weights = learn(original, edited, train.ravel())
    if weights is None:
        raise ValueError("Rank-deficient color fit")
    return apply_mapping(target, cubic_mapping(weights), .8)


def score(frame, target, truth, base, holdout, region):
    error = np.abs(frame.astype(float) - truth)
    baseline_error = np.abs(base.astype(float) - truth)
    neutral_mae = float(error[region].mean())
    baseline_neutral = float(baseline_error[region].mean())
    extra = float(error[holdout].mean() - baseline_error[holdout].mean())
    expected = (truth.astype(float) - target)[holdout]
    change = (frame.astype(float) - target)[holdout]
    norm = float(np.vdot(expected, expected))
    projection = float(np.vdot(change, expected) / norm) if norm else None
    passed = (projection is not None and .75 <= projection <= 1.25
              and neutral_mae <= min(.85 * baseline_neutral, baseline_neutral - .25)
              and extra <= 2)
    return {"neutral_mae": neutral_mae, "holdout_mae": float(error[holdout].mean()),
            "extra_mae": extra, "projection": projection, "passed": passed}


def donor_assets():
    manifest = {row["original_path"]: row for row in load(MANIFEST, HASHES[2])["files"]}
    source_by_family = defaultdict(set)
    for row in load(INDEX, HASHES[0]):
        name = row["content"]
        if name in manifest:
            source_by_family[family(name, "content")].add(name)

    @lru_cache(maxsize=32)
    def read(name):
        item = manifest[name]
        image = checked_rgb(COHORT.parent / item["prepared_path"], item["sha256"])
        return np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR)), image.shape

    return source_by_family, read


def run(out):
    rows = load(SPLIT / "private-assignment.json", SPLIT_SHA)["development"]
    source_by_family, read = donor_assets()
    archive = BASE / "reference-equivalence-development-baseline-v1"
    frozen = load(archive / "results.json", RENDER_SHA)
    yy, xx = np.indices((256, 256))
    train = (xx // 16 + yy // 16) % 2 == 0
    out.mkdir()
    results, eligible = {}, 0
    for index, row in enumerate(rows):
        target, target_shape = read(row["content"])
        region = neutral(target) & ~train
        for slot, ref in enumerate(row["refs"]):
            key = f"{index}-{slot}"
            author_path = row["different_truth" if slot == 2 else "same_truth"]
            donor_sources = source_by_family[family(ref, "reference")]
            record = {"source_available": len(donor_sources) == 1,
                      "exact_edit_number": edit_number(ref) == edit_number(author_path)}
            results[key] = record
            if not (record["source_available"] and record["exact_edit_number"]):
                continue
            eligible += 1
            donor, donor_shape = read(next(iter(donor_sources)))
            edited, edited_shape = read(ref)
            authored, authored_shape = read(author_path)
            record["donor_aligned"] = donor_shape == edited_shape and structure_score(donor, edited)["passed"]
            record["target_aligned"] = target_shape == authored_shape and structure_score(target, authored)["passed"]
            record["neutral_pixels"] = int(region.sum())
            if not (record["donor_aligned"] and record["target_aligned"] and region.sum() >= 256):
                continue
            truth = finish_spatial(target, authored, .8)
            base = checked_rgb(archive / f"{key}.png", frozen["predictions"][f"{key}.png"])
            arms = {"13": base, "donor": replay(donor, edited, target, train),
                    "own": replay(target, authored, target, train)}
            scores = {name: score(frame, target, truth, base, ~train, region) for name, frame in arms.items()}
            for name in ("donor", "own"):
                scores[name]["sha256"] = save(out / f"{key}-{name}.png", arms[name])
            record["scores"] = scores
    if len(results) != 72 or eligible != 16:
        raise ValueError("Development cohort changed")
    scored = [row["scores"] for row in results.values() if "scores" in row]
    summary = {}
    for name in ("13", "donor", "own"):
        summary[name] = {"passed": sum(row[name]["passed"] for row in scored),
                         "worst_extra_mae": max((row[name]["extra_mae"] for row in scored), default=None)}
    good = (len(scored) >= 8 and summary["donor"]["passed"] * 3 >= len(scored) * 2
            and summary["donor"]["worst_extra_mae"] <= 2)
    json_to(out / "results.json", {"split_sha256": SPLIT_SHA, "eligible_slots": eligible,
                                  "scored_slots": len(scored), "summary": summary,
                                  "passed": good, "slots": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
