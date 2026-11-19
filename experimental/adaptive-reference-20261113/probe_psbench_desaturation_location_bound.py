"""Measure an exact same-RGB global lower bound for two real local desaturation edits."""
import argparse
from pathlib import Path

import numpy as np

from probe_psbench_background_desaturation import TASKS, load, regions
from probe_sparks_shared_chroma import LUMA, render
from probe_tst_authored_neutral_intent import get_color
from probe_hdm_hdr_equal_luminance import verified_json
from probe_hdm_hdr_display import json_to
from probe_target_mirror_chroma import panel, save
from spatial_render import finish_spatial

ROOT = Path(__file__).resolve().parents[2]


def code(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.int32), -1, 0)
    return (r << 16) | (g << 8) | b


def solve(source, desired, remove, keep):
    keys = code(source)
    shared = np.intersect1d(keys[remove], keys[keep])
    use = np.isin(keys, shared) & (remove | keep)
    counts = [int((mask & use).sum()) for mask in (remove, keep)]
    if not min(counts):
        return use, len(shared), counts, None, None, None
    weight = np.where(remove[use], .5/counts[0], .5/counts[1])
    goals = desired[use]
    sorted_keys, inv = np.unique(keys[use], return_inverse=True)
    group_order = np.argsort(inv, kind="stable")
    splits = np.split(group_order, np.flatnonzero(np.diff(inv[group_order])) + 1)
    assert np.array_equal(np.unique(keys[use]), sorted_keys)
    med = np.empty_like(goals)
    normal = np.empty_like(goals)
    for indices in splits:
        for channel in range(goals.shape[1]):
            ordered = indices[np.argsort(goals[indices, channel], kind="stable")]
            index = np.searchsorted(weight[ordered].cumsum(), .5 * weight[ordered].sum())
            med[indices, channel] = goals[ordered[index], channel]
        normal[indices] = np.median(goals[indices], axis=0)
    return use, len(shared), counts, med, float((np.abs(med-goals).mean(axis=-1)*weight).sum()), float(np.abs(normal-goals).mean())


def run(folder, out, source_sha):
    base = verified_json(ROOT / "docs/research/2026-11-19-psbench-background-desaturation-evidence.json", source_sha)
    out.mkdir()
    cases = {}
    for task in TASKS:
        names = [f"{task}_{state}.jpg" for state in ("start", "end")]
        before, after = (load(folder, name, base["input_files"][name])[0] for name in names)
        truth = finish_spatial(before, after, .8)
        zones = regions(before, truth)
        remove, keep = zones["desaturated"], zones["retained"]
        assert not np.any(remove & keep)
        desired = get_color(truth)
        use, keys, count, median, fair, pooled = solve(before, desired, remove, keep)
        record = dict(shared_rgb=keys, shared_pixels=count,
                      region_pixels=[int(mask.sum()) for mask in (remove, keep)],
                      fair_bound=fair, pooled_bound=pooled,
                      eligible=min(count)>=256, excludes_both=bool(min(count)>=256 and fair > 1.5))
        if median is not None:
            light = (before @ LUMA)[use]
            result = {}
            pictures = {}
            for name, chroma in (("global", median), ("spatial", desired[use])):
                rgb = before.copy()
                rgb[use], clipped = render(chroma, light)
                err = np.abs(get_color(rgb) - desired)
                result[name] = {"sha256": save(out / f"{task}-{name}.png", rgb),
                                "mae": [float(err[use & zone].mean()) for zone in (remove, keep)],
                                "gamut_compressed_fraction": clipped}
                pictures[name] = rgb
            marked = np.rint(.45 * before).astype(np.uint8)
            marked[use & remove] = (88, 206, 244)
            marked[use & keep] = (247, 175, 83)
            record["comparison_sha256"] = panel(out / f"{task}-comparison.png", {
                "same RGB subset": {"original": before, "author": truth, "shared pixels": marked,
                                    "global replay": pictures["global"], "spatial replay": pictures["spatial"]},
            })
            record["outputs"] = result
        cases[task] = record
    json_to(out / "results.json", {"cases": cases, "source_sha256": source_sha, "strength": .8})
    print(cases)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("folder", type=Path)
    args.add_argument("out", type=Path)
    args.add_argument("source_sha")
    run(**vars(args.parse_args()))
