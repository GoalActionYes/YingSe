"""Isolate the effect of correcting warm donor images before paired fitting."""
from argparse import ArgumentParser
import json
from pathlib import Path

from probe_gros_film_b_cross import ARCHIVE, GROUPS, masks, score
from probe_gros_paired_global_local import chain
from probe_paired_cubic_cross_content import paired_fit
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save
from reference_color_app.lut import apply_mapping


def run(out, parent_sha):
    directory = BASE / "gros-paired-global-local-v1"
    parent = checked_json(directory / "results.json", parent_sha)
    start = checked_json(BASE / "gros-edit-base-coverage-v1/results.json", parent["parent"])
    limits = chain(BASE, start)
    root = BASE / "gros-film-b-cross-v2"
    film = checked_json(root / "results.json", limits["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    regions_path = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(regions_path) != original["regions_sha256"]:
        raise ValueError("Changed region ledger")
    regions = json.loads(regions_path.read_bytes())
    fitting = {}
    for donor in GROUPS:
        pair = [checked_rgb(root / f"{donor}-warm-{state}.png", film["inputs"][f"{donor}-warm"]["hashes"][state])
                for state in ("off", "on")]
        fitting[donor] = paired_fit(*pair)
    out.mkdir()
    cases = {}
    for name, prior in parent["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        generated = finish_spatial(target, apply_mapping(target, fitting[donor][0], 1), STRENGTH)
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        stored = {light: checked_rgb(directory / f"{name}-global-{light}.png", prior["hashes"][f"global-{light}"])
                  for light in ("cold", "warm")}
        area = masks(regions[person]["cold"], regions["screen"], target.shape[:2])
        states = {"cold": {"off": target, "on": stored["cold"]}, "warm": {"off": target, "on": generated}}
        readings = score(target, expected, states, area)
        for zone, comparisons in readings.items():
            if comparisons["response"]["cold"] != prior["scores"]["global"][zone]["response"]["cold"]:
                raise ValueError("Changed fixed cold-reference measurements")
        cases[name] = dict(scores={"normalized": prior["scores"]["global"], "raw": readings,
                                   "local_prior": prior["scores"]["prior"]},
                           output_sha=save(out / f"{name}-warm.png", generated),
                           panel_sha=panel(out / f"{name}-comparison.png", {
                               "target": {"original": target, "desired": expected},
                               "warm donor": {"normalized": stored["warm"], "raw": generated},
                               "cold control": {"shared": stored["cold"]}}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, cases=cases,
        fit={person: values[1] for person, values in fitting.items()},
        script_sha=digest(script), protocol_sha=digest(script.with_name("gros-paired-raw-warm-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
