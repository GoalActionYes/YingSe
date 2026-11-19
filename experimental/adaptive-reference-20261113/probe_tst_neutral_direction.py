"""Separate missing chroma magnitude from direction on authored switch points."""
import argparse
from pathlib import Path

import numpy as np

from probe_hdm_hdr_display import json_to
from probe_sparks_shared_chroma import LUMA, render
from probe_target_mirror_chroma import panel, save
from probe_tst_authored_neutral_intent import get_color
from probe_tst_neutral_switch_renders import STATES, measure
from probe_tst_same_style_error import FROZEN, frozen_cases


def endpoints(response, desire):
    wanted = np.linalg.norm(desire, axis=1)
    budget = 2 * np.linalg.norm(response, axis=1)
    projection = np.sum(response * desire, axis=1)
    valid = budget > 1e-12
    gains = np.divide(projection, (budget * .5) ** 2, out=np.zeros_like(budget), where=valid).clip(0, 2)
    factor = np.divide(budget, wanted, out=np.zeros_like(budget), where=wanted > 1e-12).clip(0, 1)
    return response * gains[:, None], desire * factor[:, None]


def candidate(target_color, truth, baseline, selected):
    original = target_color[selected]
    response = get_color(baseline)[selected] - original
    desire = get_color(truth)[selected] - original
    results, audit = {}, {}
    for name, color in zip(("length", "free"), endpoints(response, desire), strict=True):
        changed, limited = render(original + color, (baseline @ LUMA)[selected])
        output = baseline.copy()
        output[selected] = changed
        results[name] = output
        audit[name] = {"floating_mae": float(np.abs(color - desire).mean()), "limited_fraction": limited,
                       "whole_mae_delta": float(np.abs(output.astype(np.int16) - truth).mean()
                                                - np.abs(baseline.astype(np.int16) - truth).mean())}
    return results, audit


def run(out):
    out.mkdir()
    data = {}
    for key, original, authors, frames, masks, old in frozen_cases():
        selected = masks[0] | masks[1]
        base = get_color(original)
        renders = {"length": [], "free": []}
        audit = {arm: [] for arm in renders}
        for slot, (frame, state) in enumerate(zip(frames, STATES, strict=True)):
            candidates, details = candidate(base, authors[state], frame, selected)
            for name, changed in candidates.items():
                np.testing.assert_array_equal(changed[~selected], frame[~selected])
                renders[name].append(changed)
                audit[name].append(details[name] | {"sha256": save(out / f"{key}-{slot}-{name}.png", changed)})
        record = {}
        for name, arrays in renders.items():
            scores = measure(arrays, authors, masks)
            worst = max(row["whole_mae_delta"] for row in audit[name])
            record[name] = {"scores": scores, "replays": audit[name], "worst_delta": worst,
                            "passed": scores["passed"] and worst <= 2}
        comparison = panel(out / f"{key}-comparison.png", {
            "same look": {"target": original, "13": frames[0], "length": renders["length"][0],
                          "free": renders["free"][0], "author": authors[0]},
            "different look": {"target": original, "13": frames[2], "length": renders["length"][2],
                               "free": renders["free"][2], "author": authors[1]},
        })
        data[key] = {"baseline": old, "arms": record, "panel_sha256": comparison}
    if len(data) != 5:
        raise ValueError("Unexpected development cohort")
    passes = {name: sum(row["arms"][name]["passed"] for row in data.values()) for name in ("length", "free")}
    json_to(out / "results.json", {"parent_sha256": FROZEN, "cases": data, "passing_cases": passes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
