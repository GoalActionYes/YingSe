"""Audit when author outputs both protect and change near-neutral pixels."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST, load
from background_intervention import target_regions
from probe_hdm_hdr_display import json_to
from probe_sparks_shared_chroma import LUMA, chroma, render
from probe_target_mirror_chroma import checked_rgb, panel, save
from spatial_render import finish_spatial
from train_tst_reference_equivalence import BASE, SPLIT, SPLIT_SHA

GEO_SHA = "c6c71cd2ffb034908b205eb47e4cf2ee93a7349c8887d51fdccadebfca98246a"
RENDER_SHA = "44b8c73ec404d8b69d62a5399e3057d6c528c41b5241365469e673fcf1e95351"
ROOT = Path(__file__).resolve().parents[2]


def get_color(rgb):
    return chroma(rgb, rgb @ LUMA)


def proxy_masks(target, authors):
    neutral = target_regions(target, np.zeros(target.shape[:2], dtype=bool))["neutral_proxy"]
    source = get_color(target)
    a, b = map(get_color, authors)
    protect = neutral & (np.maximum(np.abs(a - source), np.abs(b - source)).max(axis=-1) <= 1)
    styled = neutral & (np.abs(b - a).max(axis=-1) >= 3)
    if np.any(protect & styled):
        raise ValueError("Protection and style groups overlap")
    return neutral, protect, styled


def different_policy_count(target, authors, styled):
    source = get_color(target)[styled]
    statuses = [np.abs(get_color(image)[styled] - source).max(axis=-1) <= 1 for image in authors]
    return int(np.count_nonzero(statuses[0] ^ statuses[1]))


def compare(images, authors, protect, styled):
    rendered = list(map(get_color, images))
    a, b = map(get_color, authors)
    errors = [np.abs(color[protect] - wanted[protect]).mean()
              for color, wanted in zip(rendered, (a, a, b), strict=True)]
    expected = (b - a)[styled]
    delta = ((rendered[2] - rendered[0]) + (rendered[2] - rendered[1]))[styled] / 2
    return {
        "protect_chroma_mae": float(np.mean(errors)),
        "styled_chroma_projection": float((delta * expected).sum() / np.square(expected).sum()),
        "styled_delta_chroma_mae": float(np.abs(delta - expected).mean()),
        "styled_under_one_code_fraction": float((np.abs(delta).max(axis=-1) <= 1).mean()),
    }


def replay(source, frames, mask):
    results, compressed = [], []
    for frame in frames:
        chosen, fraction = render(get_color(source)[mask], (frame @ LUMA)[mask])
        result = frame.copy()
        result[mask] = chosen
        results.append(result)
        compressed.append(fraction)
    return results, compressed


def run(out, include_native=False):
    baseline_dir = BASE / "reference-equivalence-development-baseline-v1"
    archive = load(baseline_dir / "results.json", RENDER_SHA)
    if not archive["metadata"]["readiness"] or archive["metadata"]["assignment_sha256"] != SPLIT_SHA:
        raise ValueError("Reference selection changed")
    geometry = load(ROOT / "docs/research/2026-11-18-tst-reference-spatial-signal-evidence.json", GEO_SHA)
    if geometry["assignment_sha256"] != SPLIT_SHA:
        raise ValueError("Geometry evidence changed")
    entries = {row["original_path"]: row for row in load(MANIFEST, HASHES[2])["files"]}
    cases = load(SPLIT / "private-assignment.json", SPLIT_SHA)["development"]

    def read(name):
        row = entries[name]
        return checked_rgb(COHORT.parent / row["prepared_path"], row["sha256"])

    out.mkdir()
    records = []
    for index, (case, alignment) in enumerate(zip(cases, geometry["cases"], strict=True)):
        if alignment["case"] != index:
            raise ValueError("Case sequence changed")
        originals = [read(case[role]) for role in ("content", "same_truth", "different_truth")]
        source, *grades = [np.array(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))
                           for rgb in originals]
        authors = [finish_spatial(source, image, .8) for image in grades]
        neutral, protect, styled = proxy_masks(source, authors)
        aligned = all(item["passed"] for item in alignment["geometry"])
        qualified = aligned and min(protect.sum(), styled.sum()) >= 256
        row = {"index": index, "aligned": aligned, "qualified": bool(qualified),
               "neutral_pixels": int(neutral.sum()), "protect_pixels": int(protect.sum()), "styled_pixels": int(styled.sum())}
        if include_native:
            native, *native_grades = originals
            native_authors = [finish_spatial(native, image, .8) for image in native_grades]
            native_neutral, native_protect, native_styled = proxy_masks(native, native_authors)
            row["one_state_protects_pixels"] = different_policy_count(source, authors, styled)
            row["native_supplement"] = {
                "neutral_pixels": int(native_neutral.sum()), "protect_pixels": int(native_protect.sum()),
                "styled_pixels": int(native_styled.sum()),
                "one_state_protects_pixels": different_policy_count(native, native_authors, native_styled),
            }
        if qualified:
            files = [f"{index}-{slot}.png" for slot in range(3)]
            original = [checked_rgb(baseline_dir / filename, archive["predictions"][filename]) for filename in files]
            all_neutrals, all_compression = replay(source, original, neutral)
            authored_protect, selective_compression = replay(source, original, protect)
            if not all(np.array_equal(current[styled], candidate[styled])
                       for current, candidate in zip(original, authored_protect, strict=True)):
                raise ValueError("Authored protection must preserve the styled region")
            arms = {"13": original, "all_neutrals": all_neutrals, "authored_protect": authored_protect}
            row["scores"] = {name: compare(frames, authors, protect, styled) for name, frames in arms.items()}
            row["compression"] = {"all_neutrals": all_compression, "authored_protect": selective_compression}
            row["file_sha256"] = {role: [save(out / f"{index}-{role}-{slot}.png", image)
                                        for slot, image in enumerate(arms[role])]
                                  for role in ("all_neutrals", "authored_protect")}
            row["comparison_sha256"] = panel(out / f"{index}-comparison.png", {
                "matched grade": {"target": source, "13": original[2], "all neutrals": all_neutrals[2],
                                  "authored protect": authored_protect[2], "author": authors[1]}})
        records.append(row)
    mixed = [row for row in records if row["qualified"]]
    improvement = sum(row["scores"]["13"]["protect_chroma_mae"]
                      - row["scores"]["authored_protect"]["protect_chroma_mae"] >= 2 for row in mixed)
    json_to(out / "results.json", {
        "assignment_sha256": SPLIT_SHA, "baseline_sha256": RENDER_SHA, "geometry_sha256": GEO_SHA,
        "cases": records, "mixed_cases": len(mixed), "protect_gain_at_least_two": improvement,
        "passed": len(mixed) >= 6 and improvement * 2 >= len(mixed),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--include-native", action="store_true")
    run(**vars(parser.parse_args()))
