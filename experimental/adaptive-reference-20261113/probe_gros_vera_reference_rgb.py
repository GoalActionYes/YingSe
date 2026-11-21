"""Change only reference channel order in the frozen VeraRetouch probe."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_gros_vera_reference_pair import ARM_NAMES, WEIGHT, load, render
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, panel, save

PARENT = "64dd323a06dfe16890b34087f250a9693341acb997e8c76ba22f43c8915340e3"
STATES = ("off", "on")


def rgb_reference_render(renderer, target, before, after):
    # The frozen author adapter reverses the reference channels; counteract only that step.
    return render(renderer, target, before[..., ::-1], after[..., ::-1])


def run(out):
    photos = BASE / "gros-film-b-cross-v2"
    archive = checked_json(photos / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    source = BASE / "gros-vera-reference-pair-v1"
    frozen = checked_json(source / "results.json", PARENT)
    case = archive["cases"]["01-from-02"]
    target = checked_rgb(photos / "01-cold-off.png", archive["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(photos / "01-from-02-expected.png", case["expected_sha"])
    area = masks(labels["01"]["cold"], labels["screen"], target.shape[:2])
    torch.set_num_threads(4)
    renderer, loaded = load(BASE / "vera-intake-v1")
    out.mkdir()
    predictions = {key: {arm: {} for arm in ARM_NAMES} for key in ("bgr", "rgb")}
    reports, boards = {}, {}
    for light in LIGHTS:
        key = f"02-{light}"
        refs = {state: checked_rgb(photos / f"{key}-{state}.png", archive["inputs"][key]["hashes"][state])
                for state in STATES}
        reports[light] = {}
        for arm, before in (("two_image", target), ("known_before", refs["off"])):
            baseline = {
                state: checked_rgb(source / f"{arm}-{light}-{state}.png", frozen["media"][light][arm][state])
                for state in STATES
            }
            candidate = {state: rgb_reference_render(renderer, target, before, after) for state, after in refs.items()}
            predictions["bgr"][arm][light] = baseline
            predictions["rgb"][arm][light] = candidate
            reports[light][arm] = {
                state: save(out / f"rgb-{arm}-{light}-{state}.png", result) for state, result in candidate.items()
            }
        boards[light] = panel(out / f"{light}-comparison.png", {
            "target": {"original": target, "Film B": expected},
            **{f"{key} {arm}": values[light] for key, arms in predictions.items() for arm, values in arms.items()},
        })
    same = rgb_reference_render(renderer, target, target, target)
    same_error = float(np.abs(same.astype(np.int16) - target).mean())
    scores = {key: {arm: score(target, expected, pictures, area) for arm, pictures in arms.items()}
              for key, arms in predictions.items()}
    if scores["bgr"] != frozen["scores"]:
        raise ValueError("Frozen BGR scores did not reproduce")
    json_to(out / "results.json", dict(parent=PARENT, weight=WEIGHT, loaded=loaded, media=reports,
                                       boards=boards, scores=scores,
                                       same=dict(rgb=dict(sha=save(out / "rgb-same.png", same), error=same_error),
                                                 bgr=frozen["same"])))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
