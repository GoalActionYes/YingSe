"""Fit a two-image chroma head with a same-reference style-difference term."""
import argparse
from pathlib import Path

import numpy as np
import torch

from audit_tst_reference_equivalence import HASHES, MANIFEST, family
from prepare_tst_reference_equivalence_split import paths
from probe_tst_donor_supervision import grade, representative
from probe_tst_neutral_direction_fit import (
    ASSAY, BASE, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA, SpatialEngine, checked_rgb,
    features, finish_spatial, get_color, images, json_to, load, masks, measure,
    neutral, save, tensor_samples, whole,
)
from probe_tst_paired_donor_edit import donor_assets, edit_number
from probe_tst_same_reference_style_pair import aligned, look_catalog, measure as style_response

ARMS = ("absolute", "delta")
FIXED = "cf08b240b708b68b69fe5309d3c8d6223e9960f32d7ee12a49b8a7e1cfc60612"


def matched(row, ref, catalog, entries):
    candidates = [catalog[family(ref, "reference"), edit_number(row[key])]
                  for key in ("same_truth", "different_truth")]
    if all(len(names) == 1 and names <= entries.keys() for names in candidates):
        return tuple(next(iter(names)) for names in candidates)
    return None


def new_sources_conflict(sources, protected):
    for a in sources:
        for b in protected:
            if (a["original_sha256"] == b["original_sha256"] or a["pixel_sha256"] == b["pixel_sha256"]
                    or (a["perceptual_hash"] ^ b["perceptual_hash"]).bit_count() <= 6):
                return True
    return False


def fit(split, entries, catalog, read, engine):
    root = representative()
    protected_paths = {p for row in split["development"] for p in paths(row)}
    for row in split["development"]:
        for ref in row["refs"]:
            pair = matched(row, ref, catalog, entries)
            if pair:
                protected_paths.update(pair)
    protected = {root(family(p, "content" if p.startswith("content_images/") else "reference"))
                 for p in protected_paths}
    protected_entries = [entries[p] for p in sorted(protected_paths)]
    x, y, dx, dy, used, dropped = [], [], [], [], [], []
    matched_count = 0
    for index, row in enumerate(split["training"]):
        original = read(row["content"])
        target = original[0]
        area = neutral(target)
        grid = np.zeros_like(area)
        grid[::4, ::4] = True
        area &= grid
        author = [read(row[key]) for key in ("same_truth", "different_truth")]
        valid = area.any() and all(aligned(original, pic) for pic in author)
        for slot, ref in enumerate(row["refs"]):
            pair = matched(row, ref, catalog, entries)
            if pair is None:
                continue
            matched_count += 1
            if (root(family(ref, "reference")) in protected or root(family(row["content"], "content")) in protected
                    or new_sources_conflict([entries[p] for p in pair], protected_entries)):
                dropped.append([index, slot, "source_conflict"])
                continue
            refs = [read(path) for path in pair]
            if not valid or not aligned(*refs):
                dropped.append([index, slot, "alignment_or_empty"])
                continue
            truth = [finish_spatial(target, state[0], .8) for state in author]
            origin = get_color(target)[area]
            labels = [get_color(image)[area] - origin for image in truth]
            design = []
            for guide, _ in refs:
                stable = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                design.append(features(target, guide, stable, area))
            x.extend(design)
            y.extend(labels)
            dx.append(design[1] - design[0])
            dy.append(labels[1] - labels[0])
            used.append([index, slot])
    if matched_count != 42:
        raise ValueError("Frozen training slots changed")
    x, y, dx, dy = map(np.concatenate, (x, y, dx, dy))
    lhs = x.T @ x + .01 * len(x) * np.eye(x.shape[1])
    rhs = x.T @ y
    coef = np.column_stack((np.linalg.solve(lhs, rhs), np.linalg.solve(lhs + dx.T @ dx, rhs + dx.T @ dy)))
    return coef, {"pairs": used, "dropped": dropped, "rows": len(x), "delta_rows": len(dx)}


def summary(cases, fixed_rows):
    data = {}
    for name in ("13", *ARMS):
        rows = [row[name] for row in cases.values()]
        pairs = [[slot[name] for slot in row] for row in fixed_rows.values()]
        data[name] = {"whole_pass": sum(row["whole"]["passed"] for row in rows),
                      "local_pass": sum(row.get("local", {}).get("passed", False) for row in rows),
                      "fixed_pass": sum(all(slot["passed"] for slot in pair) for pair in pairs),
                      "worst_extra_mae": max([row["whole"]["worst_regression"] for row in rows]
                                             + [slot["extra_mae"] for pair in pairs for slot in pair])}
    return data


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    entries = {row["original_path"]: row for row in load(MANIFEST, HASHES[2])["files"]}
    catalog = look_catalog()
    _, read = donor_assets()
    engine = SpatialEngine("multiref-v1")
    archive = BASE / "reference-equivalence-development-baseline-v1"
    stable = load(archive / "results.json", RENDER_SHA)
    fixed_dir = BASE.parent / "tst-same-reference-style-pair-v1"
    fixed = load(fixed_dir / "results.json", FIXED)
    assay = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    ready = {row["index"] for row in assay["cases"] if row["aligned"] and row["one_state_protects_pixels"] >= 256}
    if len(ready) != 5 or len(split["development"]) != 24 or engine.checkpoint_sha != stable["checkpoint_sha256"]:
        raise ValueError("Frozen sources changed")
    coefficients, training = fit(split, entries, catalog, read, engine)
    out.mkdir()
    cases, fixed_rows = {}, {}
    for index, tensors in enumerate(tensor_samples(split["development"])):
        target, refs, truth = images(tensors)
        area = neutral(target)
        expected = truth[0], truth[2]
        baseline = [checked_rgb(archive / f"{index}-{slot}.png", stable["predictions"][f"{index}-{slot}.png"])
                    for slot in range(3)]
        local = masks(target, expected) if index in ready else None

        def render_pair(inputs, frames, tag):
            states = [grade(target, guide, before, coefficients, area, ARMS)
                      for guide, before in zip(inputs, frames, strict=True)]
            hashes = {}
            for name in ARMS:
                hashes[name] = []
                for state, (sample, before) in enumerate(zip(states, frames, strict=True)):
                    after = sample[name]
                    np.testing.assert_array_equal(after[~area], before[~area])
                    hashes[name].append(save(out / f"{tag}-{name}-{state}.png", after))
            return {"13": frames, **{name: [row[name] for row in states] for name in ARMS}}, hashes

        first, hashes = render_pair(refs, baseline, f"ordinary-{index}")
        for frame in grade(target, target, baseline[0], coefficients, area, ARMS).values():
            np.testing.assert_array_equal(frame, target)
        for frame in grade(target, refs[0], baseline[0], coefficients, np.zeros_like(area), ARMS).values():
            np.testing.assert_array_equal(frame, baseline[0])
        ordinary = {name: {"whole": whole(frames, baseline, truth), "hashes": hashes.get(name),
                           **({"local": measure(frames, expected, local)} if local is not None else {})}
                    for name, frames in first.items()}
        if ordinary["13"]["whole"]["passed"] != stable["cases"][index]["passed"]:
            raise ValueError("Ordinary baseline changed")
        cases[str(index)] = ordinary
        row = split["development"][index]
        eligible_slots = fixed["cases"][str(index)]["slots"]
        for key, entry in eligible_slots.items():
            if "arms" not in entry or not entry["arms"]["13"]["scored"]:
                continue
            pair = matched(row, row["refs"][int(key)], catalog, entries)
            guides = [read(path)[0] for path in pair]
            source = [checked_rgb(fixed_dir / f"{index}-{key}-13-{side}.png", sha)
                      for side, sha in enumerate(entry["arms"]["13"]["hashes"])]
            renders, hashes = render_pair(guides, source, f"fixed-{index}-{key}")
            regions = {"whole": np.ones_like(area)}
            if area.sum() >= 256:
                regions["neutral"] = area
            scored = {}
            for name, frames in renders.items():
                scores = {label: style_response(frames, expected, region) for label, region in regions.items()}
                relevant = [score["passed"] for region in scores.values() for score in region.values() if score["scored"]]
                extra = max(float(np.abs(after.astype(float) - goal).mean() - np.abs(before.astype(float) - goal).mean())
                            for after, before, goal in zip(frames, source, expected, strict=True))
                scored[name] = {"areas": scores, "extra_mae": extra, "hashes": hashes.get(name),
                                "passed": bool(relevant) and all(relevant) and extra <= 2}
            if scored["13"]["areas"] != entry["arms"]["13"]["areas"]:
                raise ValueError("Fixed-content baseline changed")
            fixed_rows.setdefault(str(index), []).append(scored | {"slot": key})
    if len(fixed_rows) != 7 or sum(map(len, fixed_rows.values())) != 10 or len(cases) != 24:
        raise ValueError("Evaluation cohort changed")
    scores = summary(cases, fixed_rows)
    delta, control = scores["delta"], scores["absolute"]
    passed = (delta["local_pass"] >= max(3, control["local_pass"] + 2) and delta["whole_pass"] >= 7
              and delta["fixed_pass"] >= control["fixed_pass"] + 2 and delta["worst_extra_mae"] <= 2)
    json_to(out / "results.json", {"training": training, "summary": scores, "ordinary": cases, "fixed": fixed_rows,
                                  "coefficients": coefficients.tolist(), "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
