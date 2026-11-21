"""Test local RGB edge directions as a single-reference grading cue."""

from argparse import ArgumentParser
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_invariant_look import (
    descriptor, fit, guess, natural_scenes, render_style, samples, style_transforms,
)
from probe_gros_film_b_cross import ARCHIVE, GROUPS, LIGHTS, masks, score
from probe_openillumination_visible_emitters import INPUTS, PARENT, SCENES
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save

GROS = "4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260"
STYLES = ("film_a", "film_b")


def edge_directions(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    RGB = np.asarray(image, dtype=np.int16)
    values = []
    for step in (1, 4):
        delta = np.concatenate(((RGB[step:] - RGB[:-step]).reshape(-1, 3),
                                (RGB[:, step:] - RGB[:, :-step]).reshape(-1, 3)))
        sign = np.sign(delta[np.max(np.abs(delta), axis=1) >= 4])
        for a, b in combinations(sign.T, 2):
            product = a * b
            size = max(1, len(product))
            values.extend(np.count_nonzero(mask) / size
                          for mask in (product < 0, product > 0, (a == 0) ^ (b == 0)))
    return np.array(values)


def sets(transforms):
    portraits = BASE / "gros-film-b-cross-v2"
    portrait_scores = checked_json(portraits / "results.json", GROS)
    roster = checked_json(BASE / "gros-portrait-light-transfer-v1" / "results.json", ARCHIVE)
    coords = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), roster["regions_sha256"])
    views, zones = {}, {}
    for person in GROUPS:
        views[person] = {}
        for light in LIGHTS:
            key = f"{person}-{light}"
            hashes = portrait_scores["inputs"][key]["hashes"]
            off, on = [checked_rgb(portraits / f"{key}-{state}.png", hashes[state]) for state in ("off", "on")]
            views[person][light] = dict(off=off, film_a=render_style(off, transforms["film_a"]), film_b=on)
        zones[person] = masks(coords[person]["cold"], coords["screen"], views[person]["cold"]["off"].shape[:2])
    yield "portrait", views, zones, STYLES

    objects = BASE / "openillumination-fixed-prepared-v1"
    prepared = checked_json(objects / "inputs.json", INPUTS)
    checked_json(BASE / "openillumination-fixed-cross-v1" / "results.json", PARENT)
    views, zones = {}, {}
    for scene in SCENES:
        record = prepared["objects"][scene]
        views[scene] = {}
        for light, tag in (("cold", "001"), ("warm", "013")):
            off = checked_rgb(objects / f"{scene}-{tag}.png", record["frames"][tag])
            views[scene][light] = dict(off=off, film_b=render_style(off, transforms["film_b"]))
        mask_path = objects / f"{scene}-interior.png"
        if digest(mask_path) != record["mask_sha256"]:
            raise ValueError("Object mask changed")
        with Image.open(mask_path) as image:
            zone = np.array(image) > 0
        zones[scene] = {"whole": np.ones_like(zone), "object": zone}
    yield "object", views, zones, ("film_b",)


def score_case(target, desired, outputs, regions):
    return {style: score(target, desired[style],
                         {light: {"off": images["off"], "on": images[style]} for light, images in outputs.items()},
                         regions) for style in desired if style != "off"}


def run(out):
    flat = np.full((40, 40, 3), 128, np.uint8)
    ramp = flat.copy()
    ramp[:, 20:] = [140, 116, 128]
    if np.any(edge_directions(flat)) or not np.array_equal(edge_directions(ramp), np.tile([1, 0, 0, 0, 0, 1, 0, 0, 1], 2)):
        raise ValueError("Ordinal edge controls failed")

    transforms = style_transforms(BASE)
    training = natural_scenes(BASE)
    methods = {"global": descriptor, "edges": edge_directions}
    models = {arm: fit(samples(BASE, training, transforms, describe).values(), 1.)
              for arm, describe in methods.items()}
    out.mkdir()
    results = {}
    for domain, views, regions, styles in sets(transforms):
        folder = out / domain
        folder.mkdir()
        decisions = {arm: {name: {light: {grade: guess(describe(photo), models[arm])
                                           for grade, photo in grades.items()}
                                 for light, grades in lights.items()}
                           for name, lights in views.items()} for arm, describe in methods.items()}
        cases = {}
        for name, donor in permutations(views, 2):
            key = f"{name}-from-{donor}"
            origin = views[name]["cold"]["off"]
            desired = {"off": origin} | {style: finish_spatial(origin, render_style(origin, op), STRENGTH)
                                          for style, op in transforms.items() if style in styles}
            available = desired | {style: finish_spatial(origin, render_style(origin, op), STRENGTH)
                                   for style, op in transforms.items() if style not in styles}
            case, board = {}, {"target": desired}
            for arm in methods:
                renders = {light: {style: available[record[0]] for style, record in grades.items()}
                           for light, grades in decisions[arm][donor].items()}
                hashes = {f"{light}-{style}": save(folder / f"{key}-{arm}-{light}-{style}.png", rgb)
                          for light, grades in renders.items() for style, rgb in grades.items()}
                case[arm] = dict(scores=score_case(origin, desired, renders, regions[name]), hashes=hashes,
                                 exact_off=all(np.array_equal(states["off"], origin) for states in renders.values()))
                board.update({f"{arm} {light}": images for light, images in renders.items()})
            case["panel"] = panel(folder / f"{key}-comparison.png", board)
            cases[key] = case
        summary = {arm: dict(exact_off=sum(case[arm]["exact_off"] for case in cases.values()),
                             passes={style: [name for name, case in cases.items()
                                             if all(row["passed"] for row in case[arm]["scores"][style].values())]
                                     for style in styles}) for arm in methods}
        qualified = (summary["edges"]["exact_off"] == len(cases)
                     and all(len(summary["edges"]["passes"][style]) >= 5 and
                             set(summary["global"]["passes"][style]) <= set(summary["edges"]["passes"][style])
                             for style in styles))
        results[domain] = dict(decisions=decisions, cases=cases, summary=summary, qualified=qualified)
    protocol = Path(__file__).with_name("capture-ordinal-edges-protocol.md")
    json_to(out / "results.json", dict(gros=GROS, object_input=INPUTS, object_baseline=PARENT, strength=STRENGTH,
                                       protocol=digest(protocol), training={arm: model[-2:] for arm, model in models.items()},
                                       results=results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
