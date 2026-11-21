"""Try a real RGB mixture under the existing portrait off-state error budget."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_gros_rgb_mixture import (
    BASE, BASELINE_SHA, LIGHTS, REGIONS_SHA, REGRESSION_SHA, SHARED_SHA, checked_json,
    checked_rgb, digest, json_to, masks, panel, project, save, score, sources,
)

PARENT = "b810f2bcc9ed2a966464c48ae40f63ffa9f4c88564f9faab0ba6f960cf61e5a4"


def protected_scale(source, off, areas):
    delta = {light: image - source for light, image in off.items()}
    original = max(np.abs(change[area]).mean() for change in delta.values() for area in areas.values())
    cold_warm = delta["cold"] - delta["warm"]
    switching = max(np.abs(cold_warm[area]).mean() for area in areas.values())
    # Each rounded image can contribute at most 0.5 to an absolute error.
    return float(min(1., 2.5 / original if original else 1., 2. / switching if switching else 1.))


def run(out):
    root = BASE / "gros-film-b-cross-v2"
    archived = BASE / "gros-rgb-mixture-v1"
    baseline = checked_json(root / "results.json", BASELINE_SHA)
    parent = checked_json(archived / "results.json", PARENT)
    regression = checked_json(BASE / "gros-two-image-edit-regression-v1" / "results.json", REGRESSION_SHA)
    shared = checked_json(BASE / "shared-chroma-direction-v1" / "results.json", SHARED_SHA)
    coords = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    out.mkdir()
    cases = {}
    for name, record in baseline["cases"].items():
        target = name.split("-from-")[0]
        source = checked_rgb(root / f"{target}-cold-off.png", baseline["inputs"][f"{target}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{name}-expected.png", record["expected_sha"])
        signal = expected.astype(float) - source
        areas = masks(coords[target]["cold"], coords["screen"], source.shape[:2])
        pack = sources(root, record, regression["cases"][name], shared["portrait"][name], name, source)
        old = parent["cases"][name]
        control = {light: {state: checked_rgb(archived / f"{name}-{light}-{state}.png", sha)
                           for state, sha in channels.items()} for light, channels in old["images"].items()}
        if score(source, expected, control, areas) != old["scores"]:
            raise ValueError("Prior RGB mixture did not reproduce")
        moves = {light: project(pack[light]["off"], control[light]["on"] - signal) for light in LIGHTS}
        scale = protected_scale(source, moves, areas)
        floats = {light: source + scale * (image - source) for light, image in moves.items()}
        frames = {light: {"off": np.rint(image).astype(np.uint8),
                          "on": np.rint(project(pack[light]["on"], image + signal)).astype(np.uint8)}
                  for light, image in floats.items()}
        scores = score(source, expected, frames, areas)
        for area in scores.values():
            if (area["lamp_spill"]["off"] > 3 or
                    any(row["off_matches_original_mae"] > 3 for row in area["response"].values())):
                raise ValueError("Witness exceeds the original off-state budget")
        hashes = {light: {state: save(out / f"{name}-{light}-{state}.png", rgb)
                          for state, rgb in images.items()} for light, images in frames.items()}
        board = {"target": {"original": source, "expected": expected}}
        for light in LIGHTS:
            board[f"exact off / {light}"] = control[light]
            board[f"budgeted off / {light}"] = frames[light]
        cases[name] = dict(scale=scale, control=old["scores"], scores=scores, hashes=hashes,
                           board=panel(out / f"{name}-comparison.png", board))
    protocol = Path(__file__).with_name("gros-off-budget-protocol.md")
    json_to(out / "results.json", dict(parent=PARENT, protocol=digest(protocol), cases=cases))


if __name__ == "__main__":
    args = ArgumentParser(description=__doc__)
    args.add_argument("out", type=Path)
    run(args.parse_args().out)
