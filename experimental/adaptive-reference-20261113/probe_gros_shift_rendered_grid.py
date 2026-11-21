"""Move frozen bilateral-grid coefficients without changing inputs or their values."""

from argparse import ArgumentParser
from pathlib import Path

import torch

from probe_gros_film_b_cross import masks
from probe_gros_lamp_edit_near_match import measures
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, finish_spatial, json_to, panel, save,
)
from spatial_render import PreparedSpatial

PARENT = "dd2ce0592492c9c965f43f26d9d700fc8b47604dbf8c53fd4c91ab383537d9fa"
CONDITIONS = ("cold", "warm", "digital")


def shifted(prepared):
    grid = prepared.grid
    if grid.shape[-1] != 16:
        raise ValueError("Changed grid dimensions")
    moved = torch.roll(grid, 8, -1)
    torch.testing.assert_close(torch.roll(moved, 8, -1), grid, rtol=0, atol=0)
    torch.testing.assert_close(moved.sort(dim=-1).values, grid.sort(dim=-1).values, rtol=0, atol=0)
    return PreparedSpatial(prepared.engine, moved, prepared.diagnostics)


def target_grade(target, prepared):
    return finish_spatial(target, prepared.apply(target), STRENGTH)


def meets(row):
    return (row["physical_bias"] <= 3 and row["model_edit_error"] <= 3
            and .75 <= row["model_edit_projection"] <= 1.25)


def run(out):
    photos = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(photos / "results.json", BASELINE_SHA)
    previous = BASE / "gros-lamp-edit-near-match-v1"
    frozen = checked_json(previous / "results.json", PARENT)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    target = checked_rgb(photos / "01-cold-off.png", manifest["inputs"]["01-cold"]["hashes"]["off"])
    refs = {
        name: checked_rgb(photos / f"02-{name}-off.png", manifest["inputs"][f"02-{name}"]["hashes"]["off"])
        for name in ("cold", "warm")
    }
    refs["digital"] = checked_rgb(previous / "ref-digital.png", frozen["ref_hashes"]["digital"])
    ideal = checked_rgb(previous / "target-ideal.png", frozen["hashes"]["ideal"])
    area = masks(labels["01"]["cold"], labels["screen"], target.shape[:2])

    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed frozen 13 identity or strength")
    out.mkdir()
    baseline, counterfactual, hashes = {}, {}, {}
    for name in CONDITIONS:
        prepared = engine.prepare(target, refs[name])
        baseline[name] = target_grade(target, prepared)
        if save(out / f"baseline-{name}.png", baseline[name]) != frozen["hashes"][name]:
            raise ValueError(f"Frozen grade changed: {name}")
        counterfactual[name] = target_grade(target, shifted(prepared))
        hashes[name] = save(out / f"shift-{name}.png", counterfactual[name])
    original_scores = measures(target, ideal, baseline, area)
    if original_scores != frozen["target"]:
        raise ValueError("Frozen baseline measurements changed")
    changed_scores = measures(target, ideal, counterfactual, area)
    boards = panel(out / "comparison.png", {
        "target": {"original": target, "desired digital": ideal},
        "frozen": baseline, "grid shift": counterfactual,
    })
    json_to(out / "results.json", dict(
        parent=PARENT, checkpoint=CHECKPOINT, strength=STRENGTH, hashes=hashes, board=boards,
        original=original_scores, shifted=changed_scores,
        passes={arm: {name: meets(row) for name, row in scores.items()}
                for arm, scores in (("original", original_scores), ("shifted", changed_scores))},
    ))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
