"""Test palette coverage and a supported residual on raw warm-reference replays."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_arri_paired_coverage import NEIGHBORS, RADIUS
from probe_chromamill_local_fit_residual import learn_residual
from probe_gros_film_b_cross import ARCHIVE, GROUPS, masks, score
from probe_gros_paired_global_local import chain
from probe_paired_cubic_cross_content import paired_fit
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save
from reference_color_app.lut import apply_mapping


def fix(target, table, model):
    colors, inverse = np.unique(target.reshape(-1, 3), axis=0, return_inverse=True)
    tree, residual = model
    distances, indices = tree.query(colors, k=NEIGHBORS, workers=1)
    weights = 1 / np.maximum(distances, 1)
    estimate = (weights[..., None] * residual[indices]).sum(axis=1) / weights.sum(axis=1)[:, None]
    taper = np.clip((RADIUS - distances[:, -1]) / (RADIUS / 2), 0, 1)
    original = apply_mapping(colors[:, None, :], table, 1)[:, 0]
    corrected = np.rint(np.clip(original + taper[:, None] * estimate, 0, 255)).astype(np.uint8)
    output = finish_spatial(target, corrected[inverse].reshape(target.shape), STRENGTH)
    return output, distances[inverse, -1].reshape(target.shape[:2]), taper[inverse].reshape(target.shape[:2])


def run(out, parent_sha):
    raw = BASE / "gros-paired-raw-warm-v1"
    global_directory = BASE / "gros-paired-global-local-v1"
    root = BASE / "gros-film-b-cross-v2"
    prior = checked_json(raw / "results.json", parent_sha)
    global_report = checked_json(global_directory / "results.json", prior["parent"])
    upstream = checked_json(BASE / "gros-edit-base-coverage-v1/results.json", global_report["parent"])
    locks = chain(BASE, upstream)
    film = checked_json(root / "results.json", locks["film"])
    roster = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != roster["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(ledger.read_bytes())
    fitting, palette = {}, {}
    for donor in GROUPS:
        pair = [checked_rgb(root / f"{donor}-warm-{state}.png", film["inputs"][f"{donor}-warm"]["hashes"][state])
                for state in ("off", "on")]
        table, audit = paired_fit(*pair)
        if audit != prior["fit"][donor]:
            raise ValueError("Changed baseline fitting")
        fitting[donor] = table
        palette[donor] = learn_residual(*pair, table)
    out.mkdir()
    cases = {}
    for name, row in prior["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        case = film["cases"][name]
        table = fitting[donor]
        old = checked_rgb(raw / f"{name}-warm.png", row["output_sha"])
        reconstructed = finish_spatial(target, apply_mapping(target, table, 1), STRENGTH)
        np.testing.assert_array_equal(reconstructed, old)
        changed, distances, weights = fix(target, table, palette[donor])
        np.testing.assert_array_equal(changed[weights == 0], old[weights == 0])
        cold = checked_rgb(global_directory / f"{name}-global-cold.png",
                           global_report["cases"][name]["hashes"]["global-cold"])
        expected = checked_rgb(root / f"{name}-expected.png", case["expected_sha"])
        area = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        coverage = {}
        for zone, inside in area.items():
            values = distances[inside]
            coverage[zone] = dict(p50=float(np.median(values)), p95=float(np.quantile(values, .95)),
                                  eligible=float((weights[inside] > 0).mean()),
                                  mean_weight=float(weights[inside].mean()))
        arms = {key: {"cold": {"off": target, "on": cold}, "warm": {"off": target, "on": image}}
                for key, image in (("raw", old), ("residual", changed))}
        cases[name] = dict(scores={k: score(target, expected, a, area) for k, a in arms.items()},
                           coverage=coverage, image_sha=save(out / f"{name}-warm.png", changed),
                           panel_sha=panel(out / f"{name}-comparison.png", {
                               "target": {"input": target, "desired": expected},
                               "warm": {"frozen paired": old, "coverage residual": changed},
                               "cold": {"shared": cold}}))
    source = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, cases=cases,
        script_sha=digest(source), protocol_sha=digest(source.with_name("gros-paired-palette-residual-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
