"""Add same-donor cold captures to the warm paired reference color support."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_chromamill_local_fit_residual import learn_residual
from probe_gros_film_b_cross import ARCHIVE, GROUPS, masks, score
from probe_gros_paired_global_local import chain
from probe_gros_paired_palette_residual import fix
from probe_paired_cubic_cross_content import paired_fit
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save


def run(out, parent_sha):
    directory = BASE / "gros-paired-palette-residual-v1"
    prior = checked_json(directory / "results.json", parent_sha)
    raw = checked_json(BASE / "gros-paired-raw-warm-v1/results.json", prior["parent"])
    global_dir = BASE / "gros-paired-global-local-v1"
    global_fit = checked_json(global_dir / "results.json", raw["parent"])
    base = checked_json(BASE / "gros-edit-base-coverage-v1/results.json", global_fit["parent"])
    limits = chain(BASE, base)
    film_path = BASE / "gros-film-b-cross-v2"
    film = checked_json(film_path / "results.json", limits["film"])
    roster = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != roster["regions_sha256"]:
        raise ValueError("Changed region ledger")
    zones = json.loads(ledger.read_bytes())
    tables, palettes, counts = {}, {}, {}
    for donor in GROUPS:
        paired = {light: [checked_rgb(film_path / f"{donor}-{light}-{mode}.png",
                                     film["inputs"][f"{donor}-{light}"]["hashes"][mode]) for mode in ("off", "on")]
                  for light in ("warm", "cold")}
        tables[donor], report = paired_fit(*paired["warm"])
        if report != raw["fit"][donor]:
            raise ValueError("Changed warm fitting")
        same = learn_residual(*paired["warm"], tables[donor])
        merged = [np.concatenate([paired[light][i] for light in ("warm", "cold")], axis=0) for i in range(2)]
        with_cold = learn_residual(*merged, tables[donor])
        palettes[donor] = same, with_cold
        counts[donor] = dict(warm=same[0].n, pooled=with_cold[0].n)
    out.mkdir()
    cases = {}
    for name, row in prior["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(film_path / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        old = checked_rgb(directory / f"{name}-warm.png", row["image_sha"])
        warm, _ = palettes[donor]
        reconstructed, _, _ = fix(target, tables[donor], warm)
        np.testing.assert_array_equal(reconstructed, old)
        patched, distances, weights = fix(target, tables[donor], palettes[donor][1])
        np.testing.assert_array_equal(patched[weights == 0], old[weights == 0])
        cold = checked_rgb(global_dir / f"{name}-global-cold.png", global_fit["cases"][name]["hashes"]["global-cold"])
        expected = checked_rgb(film_path / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        areas = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
        coverage = {zone: dict(eligible=float((weights[m] > 0).mean()), p50=float(np.median(distances[m])),
                               mean_weight=float(weights[m].mean())) for zone, m in areas.items()}
        arms = {label: {"cold": {"off": target, "on": cold}, "warm": {"off": target, "on": img}}
                for label, img in (("warm_only", old), ("pooled", patched))}
        scores = {k: score(target, expected, value, areas) for k, value in arms.items()}
        if scores["warm_only"] != row["scores"]["residual"]:
            raise ValueError("Changed warm-only scores")
        cases[name] = dict(scores=scores, coverage={"warm_only": row["coverage"], "pooled": coverage},
                          image_sha=save(out / f"{name}-warm.png", patched),
                          panel_sha=panel(out / f"{name}-comparison.png", {
                              "target": {"original": target, "expected": expected},
                              "warm": {"warm samples": old, "two lights": patched},
                              "cold": {"shared": cold}}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, sample_sizes=counts, cases=cases,
        script_sha=digest(script), protocol_sha=digest(script.with_name("gros-paired-two-light-palette-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
