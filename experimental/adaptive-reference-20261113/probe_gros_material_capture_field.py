"""Measure reference-side material-specific correction before ordinary style rendering."""
from argparse import ArgumentParser
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch

from probe_gros_film_b_cross import ARCHIVE, GROUPS, masks, score
from probe_multilum_gray_probe import valid
from probe_physical_blue_global_fit import display, linear
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb,
    digest, finish_spatial, json_to, panel, save,
)


def anchors(regions):
    return OrderedDict(face=regions["cheek_left"] | regions["cheek_right"],
                       screen_left=regions["screen_left"], screen_right=regions["screen_right"])


def summary(image, regions):
    source = np.log(np.maximum(linear(image), 1e-7))
    supported = valid(image)
    values = []
    for area in regions.values():
        samples = source[area & supported]
        if len(samples) < 128:
            raise ValueError("Insufficient material coverage")
        values.append(np.median(samples, axis=0))
    return np.array(values)


def field(regions, correction):
    h, w = next(iter(regions.values())).shape
    y, x = np.ogrid[:h, :w]
    maps = []
    for region in regions.values():
        cy, cx = np.argwhere(region).mean(axis=0)
        distance = ((x - cx) / w) ** 2 + ((y - cy) / h) ** 2
        maps.append(np.exp(-distance / (2 * .18 ** 2)))
    weights = np.stack(maps, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ correction


def adjust(image, adjustment):
    floating = linear(image) * np.exp(adjustment)
    return display(floating), float((floating > 1).any(axis=-1).mean())


def run(out, parent_sha):
    root = BASE / "gros-film-b-cross-v2"
    parent = checked_json(root / "results.json", parent_sha)
    source = checked_json(BASE / "gros-portrait-light-transfer-v1" / "results.json", ARCHIVE)
    labels = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(labels) != source["regions_sha256"]:
        raise ValueError("Changed regions")
    coords = json.loads(labels.read_bytes())
    out.mkdir()
    arms = {}
    prepared = {}
    for person in GROUPS:
        images = {light: {state: checked_rgb(root / f"{person}-{light}-{state}.png",
                                             parent["inputs"][f"{person}-{light}"]["hashes"][state])
                          for state in ("off", "on")} for light in ("cold", "warm")}
        areas = {light: anchors(masks(coords[person][light], coords["screen"], image["off"].shape[:2]))
                 for light, image in images.items()}
        cold = summary(images["cold"]["off"], areas["cold"])
        warm = summary(images["warm"]["off"], areas["warm"])
        change = cold - warm
        maps = {"global": np.median(change, axis=0), "material": field(areas["warm"], change)}
        notes = {}
        arms[person] = {}
        for role, gain in maps.items():
            corrected, clip = {}, {}
            for state, image in images["warm"].items():
                corrected[state], clip[state] = adjust(image, gain)
            residual = np.linalg.norm(summary(corrected["off"], areas["warm"]) - cold, axis=-1)
            arms[person][role] = corrected
            notes[role] = dict(residual_log_rgb=residual.tolist(), clip=clip,
                               hashes={state: save(out / f"{person}-{role}-{state}.png", pic)
                                       for state, pic in corrected.items()})
        prepared[person] = dict(anchors=change.tolist(), before_residual=np.linalg.norm(change, axis=-1).tolist(),
                                arms=notes)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 model")
    cases = {}
    for name, row in parent["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", parent["inputs"][f"{person}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{name}-expected.png", row["expected_sha"])
        original = {light: {state: checked_rgb(root / f"{name}-13-{light}-{state}.png",
                                               row["hashes"][f"13-{light}-{state}"]) for state in ("off", "on")}
                    for light in ("cold", "warm")}
        regions = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        responses = {"13": original}
        files = {}
        for role, inputs in arms[donor].items():
            warm_results = {}
            for state, guide in inputs.items():
                frame = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
                warm_results[state] = frame
                filename = f"{name}-{role}-{state}.png"
                files[filename] = save(out / filename, frame)
            responses[role] = {"cold": original["cold"], "warm": warm_results}
        scores = {role: score(target, expected, values, regions) for role, values in responses.items()}
        board = {"target": {"input": target, "Film B": expected},
                 "cold baseline": original["cold"]} | {role: table["warm"] for role, table in responses.items()}
        cases[name] = dict(scores=scores, hashes=files, panel_sha=panel(out / f"{name}-comparison.png", board))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, checkpoint=CHECKPOINT, prepared=prepared, cases=cases,
             script_sha=digest(script), protocol_sha=digest(script.with_name("gros-material-capture-field-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
