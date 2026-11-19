"""Score known inverse edits against off-state photos from other capture sources."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from probe_capture_invariant_look import LABELS, descriptor, evaluate, natural_scenes, render_style, renderings, samples, style_transforms
from probe_capture_look_flat_residual import PARENT, PARENT_SHA
from probe_frozen_candidate_development_crosscheck import BASE
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json


def inverse_tables(transforms):
    axis = np.rint(np.linspace(0, 255, 64)).astype(np.uint8)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    tables = {}
    for name, transform in transforms.items():
        graded = render_style(lattice.reshape(512, 512, 3), transform).reshape(-1, 3)
        tables[name] = (cKDTree(graded), graded)
    return lattice, tables


def observe(rgb, lattice, tables):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    reference = np.array(image)
    flat = reference.reshape(-1, 3)
    hypotheses = {"off": (descriptor(reference), 0., reference)}
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    for name, (tree, graded) in tables.items():
        _, nearest = tree.query(colors, workers=1)
        indices = nearest[inverse]
        restored = lattice[indices].reshape(reference.shape)
        error = float(np.abs(graded[indices].astype(np.int16) - flat).mean())
        hypotheses[name] = (descriptor(restored), error, restored)
    return hypotheses


def fit_prior(groups):
    values = [sample["off"]["off"][0] for group in groups for views in group.values() for sample in views.values()]
    raw = np.asarray(values)
    return raw.mean(axis=0), np.maximum(raw.std(axis=0), 1.), len(raw)


def decide(hypotheses, prior):
    center, std, _ = prior
    errors = [hypotheses[label][1] for label in LABELS]
    cost = [float(np.sqrt(np.square((hypotheses[label][0] - center) / std).mean())) + error / 2
            for label, error in zip(LABELS, errors, strict=True)]
    order = np.argsort(cost, kind="stable")
    best, runner = order[:2]
    accepted = best != 0 and cost[runner] - cost[best] >= .10 and errors[best] <= 2
    return LABELS[best] if accepted else "off", cost


def diagnostics(cohort):
    rows = {}
    for name, views in cohort.items():
        for view, samples_by_label in views.items():
            source = samples_by_label["off"]["off"][2]
            rows[f"{name}-{view}"] = {}
            for applied in LABELS[1:]:
                _, cycle, restored = samples_by_label[applied][applied]
                edited = samples_by_label[applied]["off"][2]
                mae = lambda pixels: float(np.abs(pixels.astype(np.int16) - source).mean())
                rows[f"{name}-{view}"][applied] = {
                    "true_style_cycle_mae": cycle, "edited_to_original_mae": mae(edited),
                    "restored_to_original_mae": mae(restored),
                }
    return rows


def run(out):
    history = verified_json(BASE / PARENT / "results.json", PARENT_SHA)
    transforms = style_transforms(BASE)
    lattice, tables = inverse_tables(transforms)
    scenes = natural_scenes(BASE)
    data = samples(BASE, scenes, transforms, lambda rgb: observe(rgb, lattice, tables))
    outputs = renderings(scenes, transforms)
    out.mkdir()
    results = {}
    for domain, cohort in data.items():
        dest = out / domain
        dest.mkdir()
        prior = fit_prior([group for name, group in data.items() if name != domain])
        row = evaluate(cohort, domain, prior, outputs, dest, decide=decide)
        if row["states"] != history["arms"]["paired"][domain]["states"]:
            raise ValueError("Frozen source count changed")
        results[domain] = {**row, "prior_training_photos": prior[-1], "inverse_diagnostics": diagnostics(cohort)}
    passes = sum(row["passed"] for row in results.values())
    json_to(out / "results.json", {"historical_sha256": PARENT_SHA, "semantics": "scores are costs, lower is better",
                                   "folds": results, "passing_sources": passes, "passed": passes >= 3})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
