"""Compare physical lighting and a fitted digital edit on registered real portraits."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

from probe_gros_dinolight_spatial_residual import basis
from probe_gros_film_b_cross import error, masks
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, finish_spatial, json_to, panel, save,
)


def cheek_alignment(warm, fixed, moving):
    cold_eyes = [complex(*point) for point in fixed["eyes"]]
    warm_eyes = [complex(*point) for point in moving["eyes"]]
    factor = (warm_eyes[1] - warm_eyes[0]) / (cold_eyes[1] - cold_eyes[0])
    translated = warm_eyes[0] - factor * cold_eyes[0]
    adjusted = warm.copy()
    for label in ("cheek_left", "cheek_right"):
        x0, y0, x1, y1 = fixed[label]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        locations = factor * (xx + 1j * yy) + translated
        sx, sy = locations.real.astype(np.float32), locations.imag.astype(np.float32)
        adjusted[y0:y1, x0:x1] = cv2.remap(warm, sx, sy, cv2.INTER_LINEAR)
    return adjusted


def supported(image):
    return ((image >= 8) & (image <= 247)).all(axis=-1)


def apply(image, matrix):
    values = basis(image.astype(np.float64) / 255) @ matrix
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


def stats(cold, warm, raw_warm, digital, region, valid, held):
    chosen = region & valid & held
    return dict(
        coverage=float(np.count_nonzero(region & valid) / np.count_nonzero(region)),
        held_pixels=int(chosen.sum()),
        source_difference=error(cold[chosen], warm[chosen]),
        fit_error=error(digital[chosen], warm[chosen]),
        raw_error=error(digital[chosen], raw_warm[chosen]),
        all_pixel_error=error(digital[region], warm[region]),
    )


def measures(original, ideal, predictions, area):
    record = {}
    for label, mask in area.items():
        truth = ideal[mask].astype(float) - original[mask]
        digital = predictions["digital"][mask].astype(float) - predictions["cold"][mask]
        norm = (truth ** 2).sum()
        record[label] = dict(physical_bias=error(predictions["warm"][mask], original[mask]),
                             desired_edit=error(ideal[mask], original[mask]),
                             model_edit_projection=float((digital * truth).sum() / norm) if norm else None,
                             model_edit_error=float(np.abs(digital - truth).mean()),
                             model_output_difference=error(predictions["warm"][mask], predictions["digital"][mask]))
    return record


def run(out):
    root = BASE / "gros-film-b-cross-v2"
    inputs = checked_json(root / "results.json", BASELINE_SHA)
    ledger = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    target = checked_rgb(root / "01-cold-off.png", inputs["inputs"]["01-cold"]["hashes"]["off"])
    refs = {
        light: checked_rgb(root / f"02-{light}-off.png", inputs["inputs"][f"02-{light}"]["hashes"]["off"])
        for light in ("cold", "warm")
    }
    aligned = cheek_alignment(refs["warm"], ledger["02"]["cold"], ledger["02"]["warm"])
    model_features = basis(refs["cold"].astype(np.float64) / 255)
    valid = supported(refs["cold"]) & supported(aligned)
    y, x = np.ogrid[:valid.shape[0], :valid.shape[1]]
    even = (y // 16 + x // 16) % 2 == 0
    train = even & valid
    weights, _, rank, _ = np.linalg.lstsq(model_features[train], aligned[train] / 255, rcond=None)
    if rank != 10:
        raise ValueError("Insufficient quadratic fit rank")
    refs["digital"] = apply(refs["cold"], weights)
    zones = masks(ledger["02"]["cold"], ledger["screen"], refs["cold"].shape[:2])
    ref_scores = {name: stats(refs["cold"], aligned, refs["warm"], refs["digital"], mask, valid, ~even)
                  for name, mask in zones.items()}
    near_match = all(row["coverage"] >= .5 and row["fit_error"] <= 3 for row in ref_scores.values())
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or inputs["strength"] != STRENGTH:
        raise ValueError("Changed workbench checkpoint or strength")
    predictions = {name: finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
                   for name, guide in refs.items()}
    ideal = finish_spatial(target, apply(target, weights), STRENGTH)
    target_zones = masks(ledger["01"]["cold"], ledger["screen"], target.shape[:2])
    output_scores = measures(target, ideal, predictions, target_zones)
    out.mkdir()
    ref_hashes = {name: save(out / f"ref-{name}.png", frame)
                  for name, frame in {"digital": refs["digital"], "proxy": aligned}.items()}
    hashes = {name: save(out / f"target-{name}.png", frame)
              for name, frame in (predictions | {"ideal": ideal}).items()}
    board = panel(out / "comparison.png", {
        "reference": refs,
        "target": {"original": target, "correct digital": ideal},
        "13 outputs": predictions,
    })
    json_to(out / "results.json", dict(parent=BASELINE_SHA, checkpoint=CHECKPOINT, strength=STRENGTH,
                                       fit_pixels=int(train.sum()), coefficients=weights.tolist(),
                                       reference=ref_scores, target=output_scores, near_match=near_match,
                                       conflicting_pair=near_match and any(row["desired_edit"] > 6
                                                                          for row in output_scores.values()),
                                       ref_hashes=ref_hashes, hashes=hashes, board=board))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
