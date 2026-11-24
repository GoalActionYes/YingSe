"""Measure known-answer native-grid capacity on the two frozen warmth targets."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic

import numpy as np
import scipy
import torch

from evaluate_positional_warmth_trial import difference, heldout_scores
from probe_gros_local_grid_capacity import GRID, render_case
from probe_matched_wall_warmth_workbench import target_areas
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, digest, json_to, panel, save
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest, read_photo
from spatial_render import SpatialEngine


def nearby_region(shape, extent):
    height, width = shape[:2]
    left, top, right, bottom = extent
    y, x = np.ogrid[:height, :width]
    inside = (x >= left) & (x < right) & (y >= top) & (y < bottom)
    band = ((x >= left - width / GRID) & (x < right + width / GRID)
            & (y >= top - height / GRID) & (y < bottom + height / GRID))
    mask = band & ~inside
    if not mask.any():
        raise ValueError("The fixed nearby guard band is empty")
    return mask


def score_frame(original, expected, grade, areas, ring):
    four_states = dict(base=original, captured=original, edited=grade, captured_edited=grade)
    measured = heldout_scores(original, expected, four_states, areas)
    return dict(measured=measured, nearby_guard_mae=difference(grade, expected, ring),
                complete=all(row["passed"] for row in measured.values()))


def run(base, out):
    start = monotonic()
    if scipy.__version__ != "1.18.1":
        raise ValueError("Use the pinned SciPy capacity environment")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or STRENGTH != 0.8:
        raise ValueError("The fixed capacity renderer has changed")
    manifest = read_manifest()
    out.mkdir()
    results = {}
    for family, data in manifest["families"].items():
        if monotonic() - start >= 300:
            raise TimeoutError("Exceeded the known-answer grid budget")
        spec = data["target"]
        original = read_photo(base, spec["photos"]["off"], family)
        styled = read_photo(base, spec["photos"]["on"], family)
        areas = target_areas(original, spec, False)
        ring = nearby_region(original.shape, spec["extent"])
        fixed = {}
        grades, expected, folds, controls = render_case(engine, original, styled, areas, not results, full_frames=fixed)
        scores = {name: score_frame(original, expected, grade, areas, ring) for name, grade in grades.items()}
        full_scores = {name: score_frame(original, expected, grade, areas, ring) for name, grade in fixed.items()}
        images = {"original": original, "expected": expected, **grades, **fixed}
        tiles = {"stitched held blocks": {key: images[key] for key in ("original", "expected", *grades)},
                 "single grid full frames": fixed}
        results[family] = dict(target=spec["photos"]["off"]["scene"], shape=list(original.shape),
                               scores=scores, single_grid=full_scores, folds=folds, controls=controls,
                               nearby_pixels=int(ring.sum()),
                               sha256={key: save(out / f"{family}-{key}.png", value) for key, value in images.items()},
                               panel=panel(out / f"{family}-comparison.png", tiles))
        verdict = {key: value["complete"] for key, value in full_scores.items()}
        print(f"{family} single-grid complete: {verdict}", flush=True)
    if monotonic() - start >= 300:
        raise TimeoutError("Exceeded the known-answer grid budget")
    json_to(out / "results.json", dict(scope="offline known-answer grid capacity", manifest=MANIFEST_SHA,
                                       checkpoint=CHECKPOINT, strength=STRENGTH, scipy=scipy.__version__, cases=results))
    print(f"Grid capacity check completed in {monotonic() - start:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
