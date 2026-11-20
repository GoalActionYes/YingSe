"""Limit the existing material field to neighborhoods supported by measured regions."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from probe_gros_film_b_cross import ARCHIVE, GROUPS, masks, score
from probe_gros_material_capture_field import adjust, anchors, field, summary
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb,
    digest, finish_spatial, json_to, panel, save,
)


def support(shape, rectangles):
    y, x = np.ogrid[:shape[0], :shape[1]]
    result = np.zeros(shape, dtype=float)
    for x0, y0, x1, y1 in rectangles:
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError("Invalid support rectangle")
        dx = np.maximum(np.maximum(x0 - x, x - (x1 - 1)), 0) * (2. / (x1 - x0))
        dy = np.maximum(np.maximum(y0 - y, y - (y1 - 1)), 0) * (2. / (y1 - y0))
        distance = np.hypot(dx, dy)
        weight = np.where(distance < 1, (1 + np.cos(np.pi * np.minimum(distance, 1))) / 2, 0.)
        np.maximum(result, weight, out=result)
    return result


def run(out, parent_sha):
    previous = BASE / "gros-material-capture-field-v1"
    parent = checked_json(previous / "results.json", parent_sha)
    root = BASE / "gros-film-b-cross-v2"
    film = checked_json(root / "results.json", parent["parent"])
    source = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != source["regions_sha256"]:
        raise ValueError("Changed scoring regions")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    corrected, prepared = {}, {}
    for person in GROUPS:
        images = {light: {state: checked_rgb(root / f"{person}-{light}-{state}.png",
                                             film["inputs"][f"{person}-{light}"]["hashes"][state])
                          for state in ("off", "on")} for light in ("cold", "warm")}
        shape = images["warm"]["off"].shape[:2]
        bounds = coords[person]["warm"]
        rectangles = [bounds[key] for key in ("cheek_left", "cheek_right")] + list(coords["screen"].values())
        weight = support(shape, rectangles)
        cold_regions, warm_regions = (anchors(masks(coords[person][light], coords["screen"], shape))
                                      for light in ("cold", "warm"))
        cold_rgb = summary(images["cold"]["off"], cold_regions)
        change = np.asarray(parent["prepared"][person]["anchors"])
        fixed = np.median(change, axis=0)
        original_field = field(warm_regions, change)
        gain = fixed + weight[..., None] * (original_field - fixed)
        corrected[person] = {}
        hashes, clipping = {}, {}
        for state, image in images["warm"].items():
            result, clipping[state] = adjust(image, gain)
            index = parent["prepared"][person]["arms"]["global"]["hashes"][state]
            global_frame = checked_rgb(previous / f"{person}-global-{state}.png", index)
            if not np.array_equal(result[weight == 0], global_frame[weight == 0]):
                raise ValueError("Correction changed unsupported pixels")
            corrected[person][state] = result
            hashes[state] = save(out / f"{person}-bounded-{state}.png", result)
        remain = np.linalg.norm(summary(corrected[person]["off"], warm_regions) - cold_rgb, axis=-1)
        prepared[person] = dict(hashes=hashes, clip=clipping, support_fraction=float((weight > 0).mean()),
                                full_weight_fraction=float((weight == 1).mean()), residual_log_rgb=remain.tolist())
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 checkpoint")
    cases = {}
    for name, case in film["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{name}-expected.png", case["expected_sha"])
        cold = {state: checked_rgb(root / f"{name}-13-cold-{state}.png", case["hashes"][f"13-cold-{state}"])
                for state in ("off", "on")}
        warm = {state: finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
                for state, ref in corrected[donor].items()}
        regions = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        summaries = parent["cases"][name]
        scores = summaries["scores"] | {"bounded": score(target, expected, {"cold": cold, "warm": warm}, regions)}
        boards = {"target": {"original": target, "desired": expected}, "cold reference": cold}
        for arm in ("global", "material"):
            boards[arm] = {state: checked_rgb(previous / f"{name}-{arm}-{state}.png",
                                              summaries["hashes"][f"{name}-{arm}-{state}.png"])
                           for state in ("off", "on")}
        boards["bounded"] = warm
        cases[name] = dict(scores=scores, hashes={state: save(out / f"{name}-{state}.png", frame)
                                                   for state, frame in warm.items()},
                           panel_sha=panel(out / f"{name}-comparison.png", boards))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, film=parent["parent"], prepared=prepared, cases=cases,
             script_sha=digest(script), protocol_sha=digest(script.with_name("gros-bounded-material-field-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
