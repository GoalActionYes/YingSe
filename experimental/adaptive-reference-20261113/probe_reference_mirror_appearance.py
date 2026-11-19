"""Fixed reference-appearance reflection diagnostic on the four frozen suites."""
import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_frozen_candidate_development_crosscheck import (
    BASE, CHECKPOINT, COUNTS, MANIFESTS, STRENGTH, WEIGHT, WEIGHT_SHA256,
    background, chart, digest, facd, film_b, json_to, verify,
)
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


def grade(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True}
    content, stats = engine.analysis(target), engine.analysis(reference)
    # Keep the identical statistics tensor in both arms, including its preprocessing.
    mirrored = engine.analysis(reference[:, ::-1].copy())
    images = []
    for appearance in (stats, mirrored):
        grid = appearance_grid(engine.model, engine.api, content, stats, appearance_reference=appearance)
        rgb = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)
        images.append(rgb.astype(np.int16))
    original, flipped = images
    result = np.rint((original + flipped) * .5).astype(np.uint8)
    delta = np.abs(flipped - original)
    return result, {
        "mirror_delta_mae": float(delta.mean()),
        "mirror_delta_max": int(delta.max()),
        "mean_delta_from_expanded": float(np.abs(result.astype(np.int16) - original).mean()),
        "mean_delta_from_13": float(np.abs(result.astype(np.int16) - baseline).mean()),
    }


@torch.inference_mode()
def run(out):
    verified = verify(BASE, COUNTS)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Unexpected expanded checkpoint")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected base checkpoint")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    sample = np.array([[[4, 20, 200], [255, 0, 128]]], dtype=np.uint8)
    np.testing.assert_array_equal(grade(engine, sample, sample.copy(), 255 - sample)[0], sample)
    out.mkdir()
    suites = {key: evaluate(engine, out / key, grade, "Appearance reflection mean")
              for key, evaluate in (("background", background), ("facd", facd), ("film_b", film_b), ("chart", chart))}
    json_to(out / "results.json", {"baseline_sha256": CHECKPOINT, "candidate_sha256": WEIGHT_SHA256,
            "strength": STRENGTH, "inputs": MANIFESTS, "preflight_counts": verified, "suites": suites})
    print({key: {k: v for k, v in suite.items() if k.endswith("passed") or k.endswith("cases") and k != "cases"}
           for key, suite in suites.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
