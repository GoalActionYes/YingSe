"""Transfer donor-only Film B grids through the unchanged spatial renderer."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_grid_expression import render_grid, rendering_features, solve_grid
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)


def run(out):
    archive = BASE / "gros-film-b-cross-v2"
    frozen = checked_json(archive / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed frozen baseline")

    out.mkdir()
    pairs, fits, depth_maps, fit_audits = {}, {}, {}, {}
    for key, record in frozen["inputs"].items():
        original, edited = (
            checked_rgb(archive / f"{key}-{state}.png", record["hashes"][state])
            for state in ("off", "on")
        )
        pairs[key] = (original, edited)
        affine, depth, inverse = rendering_features(engine, original, edited)
        depth_maps[key] = depth
        y, x = np.ogrid[:original.shape[0], :original.shape[1]]
        train = (y // 16 + x // 16) % 2 == 0
        fits[key], fit_audits[key] = {}, {}
        for arm, weights in (("shared", np.ones_like(depth[..., :1])), ("depth", depth)):
            coefficients, rank = solve_grid(affine, weights, inverse, train)
            fits[key][arm] = coefficients
            fit_audits[key][arm] = dict(
                rank=int(rank), active_layers=np.flatnonzero((weights[train] > 1e-5).any(axis=0)).tolist(),
            )

    cases = {}
    for person, donor in permutations(GROUPS, 2):
        target, edited = pairs[f"{person}-cold"]
        expected = finish_spatial(target, edited, STRENGTH)
        regions = masks(labels[person]["cold"], labels["screen"], target.shape[:2])
        slug = f"{person}-from-{donor}"
        results, boards = {}, {"target": {"original": target, "expected": expected}}
        for arm in ("shared", "depth"):
            predictions, hashes, unsupported = {}, {}, {}
            for light in LIGHTS:
                key = f"{donor}-{light}"
                prepared = render_grid(engine, target, fits[key][arm])
                grade = finish_spatial(target, prepared.apply(target), STRENGTH)
                predictions[light] = {"off": target, "on": grade}
                hashes[light] = save(out / f"{slug}-{arm}-{light}.png", grade)
                if arm == "depth":
                    missing = np.ones(8, dtype=bool)
                    missing[fit_audits[key][arm]["active_layers"]] = False
                    mass = depth_maps[f"{person}-cold"][..., missing].sum(-1)
                    unsupported[light] = {name: float(mass[mask].mean()) for name, mask in regions.items()}
            measured = score(target, expected, predictions, regions)
            results[arm] = dict(scores=measured, hashes=hashes, unsupported_depth_mass=unsupported,
                                passed=all(row["passed"] for row in measured.values()))
            boards[arm] = {light: pair["on"] for light, pair in predictions.items()}
        cases[slug] = results | {"board": panel(out / f"{slug}-comparison.png", boards)}
    protocol = Path(__file__).with_name("gros-cross-grid-expression-protocol.md")
    json_to(out / "results.json", dict(parent=BASELINE_SHA, regions=REGIONS_SHA, checkpoint=CHECKPOINT,
                                       strength=STRENGTH, protocol=digest(protocol), fits=fit_audits, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
