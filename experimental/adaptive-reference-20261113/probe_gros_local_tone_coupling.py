"""Compare local tone/chroma coupling against a global descriptor on held-out portraits."""
from argparse import ArgumentParser
from io import BytesIO
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_invariant_look import (
    descriptor, fit, guess, natural_scenes, render_style, samples, style_transforms,
)
from probe_gros_film_b_cross import ARCHIVE, GROUPS, LIGHTS, masks, score
from probe_multilum_gray_probe import chroma, valid
from probe_physical_blue_global_fit import linear
from probe_target_mirror_chroma import BASE, LUMA, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save


def describe(image):
    scaled = Image.fromarray(image)
    scaled.thumbnail((256, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(scaled)
    y = np.log(np.maximum(linear(rgb) @ LUMA, 1e-7))
    cb = chroma(rgb)
    support = valid(rgb)
    values = []
    for iy in range(0, rgb.shape[0] - 15, 16):
        for ix in range(0, rgb.shape[1] - 15, 16):
            block = np.s_[iy:iy + 16, ix:ix + 16]
            kept = support[block]
            if kept.sum() < 128:
                continue
            tone, color = y[block][kept], cb[block][kept]
            tone -= tone.mean()
            color -= color.mean(axis=0)
            spreads = np.sqrt((tone @ tone) * np.sum(np.square(color), axis=0))
            if np.any(spreads / len(tone) <= .0001):
                continue
            values.append(tone @ color / spreads)
    return np.quantile(values, [.25, .5, .75], axis=0).ravel() if len(values) >= 8 else np.zeros(6)


def jpeg(rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    with Image.open(buffer) as loaded:
        return np.array(loaded.convert("RGB"))


def run(out, parent_sha):
    known = style_transforms(BASE)
    archive = BASE / "gros-film-b-cross-v2"
    parent = checked_json(archive / "results.json", parent_sha)
    roster = checked_json(BASE / "gros-portrait-light-transfer-v1" / "results.json", ARCHIVE)
    regions = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(regions) != roster["regions_sha256"]:
        raise ValueError("Changed saved scoring areas")
    zones = json.loads(regions.read_bytes())
    models = {}
    for label, build in (("global", descriptor), ("local", describe)):
        training = samples(BASE, natural_scenes(BASE), known, describe=build)
        models[label] = (build, fit(training.values(), 1.))
    out.mkdir()
    people = {}
    for person in GROUPS:
        people[person] = {}
        for light in LIGHTS:
            key = f"{person}-{light}"
            record = parent["inputs"][key]
            if not record["qualified"]:
                raise ValueError("Previously unqualified reference")
            off, film_b = [checked_rgb(archive / f"{key}-{role}.png", record["hashes"][role]) for role in ("off", "on")]
            people[person][light] = dict(off=off, film_a=render_style(off, known["film_a"]), film_b=film_b)
    all_results = {}
    for encoding, recode in (("original", np.asarray), ("jpeg85", jpeg)):
        directory = out / encoding
        directory.mkdir()
        decisions, predictions = {}, {}
        for label, (build, model) in models.items():
            decisions[label] = {}
            for person, lights in people.items():
                decisions[label][person] = {light: {style: guess(build(recode(rgb)), model)
                                                    for style, rgb in states.items()} for light, states in lights.items()}
            cases = {}
            for person, donor in permutations(GROUPS, 2):
                target = people[person]["cold"]["off"]
                area = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
                renderings = {"off": target} | {style: finish_spatial(target, render_style(target, operation), STRENGTH)
                                                for style, operation in known.items()}
                outputs = {light: {style: renderings[choice[0]] for style, choice in states.items()}
                           for light, states in decisions[label][donor].items()}
                reports = {}
                for style in ("film_a", "film_b"):
                    exchange = {light: {"off": items["off"], "on": items[style]} for light, items in outputs.items()}
                    reports[style] = score(target, renderings[style], exchange, area)
                slug = f"{label}-{person}-from-{donor}"
                hashes = {f"{light}-{style}": save(directory / f"{slug}-{light}-{style}.png", image)
                          for light, items in outputs.items() for style, image in items.items()}
                cases[slug] = dict(scores=reports, hashes=hashes, panel_sha=panel(directory / f"{slug}-comparison.png",
                    {"target": renderings, "cold": outputs["cold"], "warm": outputs["warm"]}))
            predictions[label] = cases
        all_results[encoding] = dict(decisions=decisions, predictions=predictions)
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, encodings=all_results,
             script_sha=digest(script), protocol_sha=digest(script.with_name("gros-local-tone-coupling-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
