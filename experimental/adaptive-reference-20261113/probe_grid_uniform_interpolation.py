"""Check equal-weight interpolation of the frozen spatial color grid."""
import argparse
from pathlib import Path

import numpy as np
import torch

from audit_tst_reference_equivalence import HASHES, MANIFEST, family
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH, background, chart, facd, film_b
from probe_hdm_hdr_display import json_to
from probe_tst_fixed_content_delta_supervision import FIXED, matched
from probe_tst_neutral_direction_fit import BASE as TST_BASE, SPLIT, SPLIT_SHA, checked_rgb, load, neutral, save
from probe_tst_paired_donor_edit import donor_assets
from probe_tst_same_reference_style_pair import look_catalog, measure
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from verify_development_inputs import verify


def uniform_grid(grid):
    alpha = grid[:, -1:]
    if torch.any(alpha <= 0):
        raise ValueError("Expected positive spatial weights")
    return torch.cat((grid[:, :-1] / alpha, torch.ones_like(alpha)), dim=1)


def compare(engine, target, ref, baseline):
    if np.array_equal(target, ref):
        return target.copy(), {"identity": True}
    prepared = engine.prepare(target, ref)
    control = finish_spatial(target, prepared.apply(target), STRENGTH)
    np.testing.assert_array_equal(control, baseline)
    candidate = PreparedSpatial(engine, uniform_grid(prepared.grid), {})
    output = finish_spatial(target, candidate.apply(target), STRENGTH)
    weights = prepared.grid[:, -1]
    return output, {"control_matches": True, "alpha_min": float(weights.min()), "alpha_max": float(weights.max()),
                    "change_rgb8": float(np.abs(output.astype(np.int16) - control).mean())}


def toy():
    grid = torch.tensor([[[0., 4.], [1., 4.]]])
    unweighted = uniform_grid(grid).mean(-1)
    control = grid.mean(-1)
    np.testing.assert_array_equal(unweighted[:, 0] / unweighted[:, 1], np.array([.5]))
    np.testing.assert_allclose(control[:, 0] / control[:, 1], np.array([.8]))
    constant = torch.tensor([[[4., 8.], [2., 2.]]])
    np.testing.assert_array_equal(uniform_grid(constant).mean(-1)[:, 0], constant.mean(-1)[:, 0] / 2)


def fixed_reference(engine, out):
    out.mkdir()
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)["development"]
    entries = {row["original_path"]: row for row in load(MANIFEST, HASHES[2])["files"]}
    catalog = look_catalog()
    _, read = donor_assets()
    prior = TST_BASE.parent / "tst-same-reference-style-pair-v1"
    frozen = load(prior / "results.json", FIXED)
    groups = {}
    for key, row in frozen["cases"].items():
        index = int(key)
        assignment = split[index]
        target = read(assignment["content"])[0]
        author = [finish_spatial(target, read(assignment[name])[0], STRENGTH)
                  for name in ("same_truth", "different_truth")]
        area = neutral(target)
        regions = {"whole": np.ones_like(area)}
        if area.sum() >= 256:
            regions["neutral"] = area
        for slot, entry in row["slots"].items():
            if "arms" not in entry or not entry["arms"]["13"]["scored"]:
                continue
            references = matched(assignment, assignment["refs"][int(slot)], catalog, entries)
            source = [checked_rgb(prior / f"{index}-{slot}-13-{i}.png", sha)
                      for i, sha in enumerate(entry["arms"]["13"]["hashes"])]
            outputs, diagnostics = [], []
            for i, (reference, baseline) in enumerate(zip(references, source, strict=True)):
                image, detail = compare(engine, target, read(reference)[0], baseline)
                output = out / f"{index}-{slot}-{i}.png"
                details = detail | {"sha256": save(output, image)}
                outputs.append(image)
                diagnostics.append(details)
            scores = {key: measure(outputs, author, region) for key, region in regions.items()}
            extra = max(float(np.abs(after.astype(float) - desired).mean() - np.abs(before.astype(float) - desired).mean())
                        for after, before, desired in zip(outputs, source, author, strict=True))
            passed = extra <= 2 and all(value["passed"] for area in scores.values()
                                        for value in area.values() if value["scored"])
            groups.setdefault(key, {})[slot] = {"scores": scores, "max_added_mae": extra,
                                              "passed": passed, "diagnostics": diagnostics}
    if len(groups) != 7 or sum(map(len, groups.values())) != 10:
        raise ValueError("Fixed reference cohort changed")
    passes = sum(all(item["passed"] for item in group.values()) for group in groups.values())
    worst = max(item["max_added_mae"] for group in groups.values() for item in group.values())
    return {"cases": groups, "passing_cases": passes, "max_added_mae": worst,
            "passed": passes >= frozen["summary"]["13"]["groups_pass"] + 2 and worst <= 2}


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    toy()
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Baseline weights changed")
    sample = np.array([[[4, 20, 200], [255, 0, 128]]], dtype=np.uint8)
    identity, _ = compare(engine, sample, sample.copy(), 255 - sample)
    np.testing.assert_array_equal(identity, sample)
    out.mkdir()
    suites = {name: evaluator(engine, out / name, compare, "Uniform grid weights")
              for name, evaluator in (("background", background), ("facd", facd), ("film_b", film_b), ("chart", chart))}
    suites["fixed_reference"] = fixed_reference(engine, out / "fixed_reference")
    json_to(out / "results.json", {"checkpoint_sha256": CHECKPOINT, "fixed_parent_sha256": FIXED,
                                  "suites": suites, "passed": all(row["passed"] for row in suites.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
