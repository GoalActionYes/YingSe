"""Check whether the pinned spatial filter suppresses useful style response."""

import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_frozen_candidate_development_crosscheck import (
    BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256, background, chart, facd, film_b,
)
from probe_frozen_render_blend_bound import FULL, FULL_SHA
from probe_frozen_render_clip_order import pixel_sha, verify_control
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from verify_development_inputs import verify


def compare(engine, target, reference, _baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True, "control_pixels_sha256": pixel_sha(target)}
    api = dict(engine.api, lowpass=lambda grid: grid)
    before = appearance_grid(engine.model, api, engine.analysis(target), engine.analysis(reference))
    after = engine.api["lowpass"](before)
    control, candidate = (
        finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)
        for grid in (after, before)
    )
    difference = np.abs(candidate.astype(np.int16) - control)
    return candidate, {
        "control_pixels_sha256": pixel_sha(control),
        "no_filter_vs_control_mae_rgb8": float(difference.mean()),
        "no_filter_vs_control_max_rgb8": int(difference.max()),
    }


def verify_controls(suites, original):
    verify_control({name: suites[name] for name in ("background", "film_b")}, original)
    for suite in ("facd", "chart"):
        new = suites[suite]
        old = original["suites"][suite]
        if suite == "chart":
            new, old = {".": new}, {".": old}
        else:
            new, old = new["cases"], old["cases"]
        for group, row in new.items():
            for role, detail in row["outputs"].items():
                filename = f"{role}.png" if suite == "chart" else role
                path = BASE / FULL / suite / group / filename
                archive = checked_rgb(path, old[group]["outputs"][role]["sha256"])
                if pixel_sha(archive) != detail["control_pixels_sha256"]:
                    raise ValueError(f"Frozen control pixels changed: {path}")


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    parent = verified_json(BASE / FULL / "results.json", FULL_SHA)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Expanded checkpoint changed")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    same = np.array([[[0, 10, 200], [255, 216, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(compare(engine, same, same.copy(), None)[0], same)
    out.mkdir()
    suites = {name: evaluator(engine, out / name, compare, "without XY smoothing")
              for name, evaluator in (("background", background), ("facd", facd), ("film_b", film_b), ("chart", chart))}
    verify_controls(suites, parent)
    json_to(out / "results.json", {
        "parent_sha256": FULL_SHA, "checkpoint_sha256": WEIGHT_SHA256, "strength": STRENGTH,
        "identity_passed": True, "suites": suites, "passed": all(row["passed"] for row in suites.values()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
