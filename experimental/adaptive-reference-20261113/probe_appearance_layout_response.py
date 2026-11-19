"""Compare the frozen expanded model with and without reference appearance."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from background_intervention import target_regions
from pairwise_context import pairwise_histograms
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import digest, metrics
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_mls_film_light_cross import SOURCE_SHA
from probe_reference_layout_collision import PARENT_SHA, distance, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


LAYOUT_SHA = "7f7e45043bf90a45257cf5dbf73c4cb453fb01a782552081dbee936abc7c0bd1"
SIDES = ("left", "right")
ARMS = ("stats", "full")


@torch.inference_mode()
def run(base, out):
    source = base / "mls-linear-pilot"
    captures = verified_json(source / "results.json", SOURCE_SHA)
    parent = base / "mls-film-light-cross-v1"
    crossed = verified_json(parent / "results.json", PARENT_SHA)
    prior = base / "reference-layout-collision-v1"
    frozen = verified_json(prior / "results.json", LAYOUT_SHA)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Changed frozen expanded checkpoint")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    cases = {}
    for scene, case in frozen["cases"].items():
        filename = f"{scene}_2HAL_DESK.png"
        target = verified(source / filename, captures["scenes"][scene]["inputs_sha256"][filename])
        refs = {side: verified(prior / f"{scene}-{side}-reference.png", case["image_sha256"][side]["reference"])
                for side in SIDES}
        content = engine.analysis(target)
        guides = {side: engine.analysis(rgb) for side, rgb in refs.items()}
        colors = {side: engine.api["srgb_to_ycbcr"](guide) for side, guide in guides.items()}
        for histogram in (engine.api["batch_histogram_1D"], pairwise_histograms):
            torch.testing.assert_close(histogram(colors["left"]), histogram(colors["right"]), atol=0, rtol=0)

        arms = {}
        for arm in ARMS:
            grids = {side: appearance_grid(engine.model, engine.api, content, guide,
                                           disable_appearance=arm == "stats") for side, guide in guides.items()}
            hashes = {}
            for side, grid in grids.items():
                output = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8)
                hashes[side] = save(out / f"{scene}-{arm}-{side}.png", output)
            arms[arm] = {"grid_max_difference": float((grids["left"] - grids["right"]).abs().max()),
                         "hashes": hashes}

        authored = verified(parent / f"{scene}-standard-reference-film.png", crossed["references"][scene]["standard"])
        answers = {side: verified(prior / f"{scene}-{side}-expected.png", case["image_sha256"][side]["expected"])
                   for side in SIDES}
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(image, ~whole)["neutral_proxy"] for image in (target, authored)))
        if neutral.sum() != crossed["cases"][scene]["neutral_pixels"]:
            raise ValueError("Changed scored neutral support")
        for arm, report in arms.items():
            outputs = {side: verified(out / f"{scene}-{arm}-{side}.png", report["hashes"][side]) for side in SIDES}
            report["regions"] = {}
            for label, region in (("whole", whole), ("neutral", neutral)):
                response = metrics(*answers.values(), *outputs.values(), region)
                report["regions"][label] = {
                    "output_difference_mae": distance(outputs["left"][region], outputs["right"][region]),
                    "response_projection": response["projection"], "response_mae": response["response_mae"],
                    "errors": {side: distance(outputs[side][region], answers[side][region]) for side in SIDES},
                    "passed": response["passed"] and max(response["off_matches_original_mae"],
                                                         response["on_matches_authored_mae"]) <= 3,
                }
            if arm == "full":
                report["panel_sha256"] = panel(out / f"{scene}-comparison.png", refs, target, answers, outputs,
                                               output_label="Expanded output")
        if arms["stats"]["grid_max_difference"] != 0 or len(set(arms["stats"]["hashes"].values())) != 1:
            raise ValueError("Pure statistics unexpectedly responded to layout")
        cases[scene] = arms
    passed = all(row["passed"] for case in cases.values() for row in case["full"]["regions"].values())
    json_to(out / "results.json", {"layout_sha256": LAYOUT_SHA, "weight_sha256": WEIGHT_SHA256,
                                   "strength": .8, "cases": cases, "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
