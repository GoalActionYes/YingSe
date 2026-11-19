"""Separate the spatial decision from desaturation magnitude on real edits."""
import argparse
from pathlib import Path

import numpy as np

from probe_psbench_background_desaturation import TASKS, REC709, load, regions, score
from probe_hdm_hdr_equal_luminance import verified_json
from probe_hdm_hdr_display import json_to
from probe_target_mirror_chroma import checked_rgb, panel, save
from spatial_render import finish_spatial

ROOT = Path(__file__).resolve().parents[2]


def retained_amount(source, authored, mask):
    x = source.astype(np.float64) - (source @ REC709)[..., None]
    y = authored.astype(np.float64) - (authored @ REC709)[..., None]
    x, y = (image[mask] for image in (x, y))
    if not np.any(x):
        raise ValueError("No chroma in authoring area")
    return float(np.clip(np.sum(x * y) / np.sum(x*x), 0, 1))


def replay(source, area, amount):
    light = (source @ REC709)[area, None]
    result = source.copy()
    result[area] = np.rint(light + (source[area] - light) * amount).astype(np.uint8)
    return result


def run(media, out, identity):
    base = verified_json(ROOT / "docs/research/2026-11-19-psbench-background-desaturation-evidence.json", identity)
    records = {}
    for task in TASKS:
        keys = [f"{task}_{role}.jpg" for role in ("start", "end")]
        before, after = (load(media, k, base["input_files"][k])[0] for k in keys)
        truth = finish_spatial(before, after, .8)
        area = regions(before, truth)
        assert area["desaturated"].sum() >= 256 and area["retained"].sum() >= 256
        assert not np.any(area["desaturated"] & area["retained"])
        records[task] = before, truth, area, retained_amount(before, truth, area["desaturated"])
    out.mkdir()
    cases = {}
    for task, donor in (TASKS, TASKS[::-1]):
        before, truth, area, oracle = records[task]
        observed = {}
        board = {"original": before, "author": truth}
        arm_amount = {"fixed": .2, "from_reference": records[donor][3]}
        for arm, amount in arm_amount.items():
            for_output = replay(before, area["desaturated"], amount)
            np.testing.assert_array_equal(for_output[~area["desaturated"]], before[~area["desaturated"]])
            peak = float(np.abs((for_output.astype(float) - before) @ REC709).max())
            assert peak <= .5 + 1e-9
            measured = score(before, truth, before, for_output, area)
            path = out / f"{task}-{arm}.png"
            observed[arm] = {"retained_amount": amount, "peak_luma_error": peak,
                             "scores": measured, "sha256": save(path, for_output)}
            board[arm] = for_output
        old_folder = media.parent / "psbench-background-desaturation-v1"
        off, on = (checked_rgb(old_folder / f"{task}-{state}.png", base["results"][task]["hashes"][state])
                   for state in ("off", "on"))
        cases[task] = {"reference": donor, "oracle_amount": oracle, "arms": observed,
                      "frozen_13": score(before, truth, off, on, area),
                      "comparison_sha256": panel(out / f"{task}-comparison.png", {"offline regions": board})}
    json_to(out / "results.json", {"source_sha256": identity, "cases": cases})
    for task, case in cases.items():
        print(task, case["oracle_amount"], {
            k: (v["scores"]["passed"], v["scores"]["regions"]["desaturated"]["response_mae"],
                v["scores"]["regions"]["desaturated"]["projection"])
            for k,v in case["arms"].items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("identity")
    run(**vars(parser.parse_args()))
