"""Render a continuous mixture from frozen single-reference style scores."""
from argparse import ArgumentParser
import json
from itertools import permutations
from pathlib import Path

import numpy as np

from probe_capture_invariant_look import LABELS, fit, guess, natural_scenes, render_style, samples, style_transforms
from probe_gros_film_b_cross import ARCHIVE, GROUPS, LIGHTS, masks, score
from probe_gros_local_tone_coupling import describe, jpeg
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save


def fractions(scores):
    positive = np.maximum(scores, 0)
    mass = float(positive.sum())
    if mass == 0:
        return np.array([1., 0., 0.])
    return positive / mass


def mixture(frames, scores):
    weights = fractions(scores)
    stacked = np.stack([frames[label] for label in LABELS], axis=-1)
    return np.rint(np.clip(stacked @ weights, 0, 255)).astype(np.uint8)


def run(out, parent_sha):
    frozen = BASE / "gros-local-tone-coupling-v1"
    prior = checked_json(frozen / "results.json", parent_sha)
    archive = BASE / "gros-film-b-cross-v2"
    source = checked_json(archive / "results.json", prior["parent"])
    roster = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != roster["regions_sha256"]:
        raise ValueError("Changed region ledger")
    zones = json.loads(ledger.read_bytes())
    ops = style_transforms(BASE)
    training = samples(BASE, natural_scenes(BASE), ops, describe=describe)
    model = fit(training.values(), 1.)
    people = {}
    for person in GROUPS:
        people[person] = {}
        for light in LIGHTS:
            entry = source["inputs"][f"{person}-{light}"]["hashes"]
            start, graded = [checked_rgb(archive / f"{person}-{light}-{key}.png", entry[key]) for key in ("off", "on")]
            people[person][light] = dict(off=start, film_a=render_style(start, ops["film_a"]), film_b=graded)
    out.mkdir()
    suite = {}
    for kind, recode in (("original", np.asarray), ("jpeg85", jpeg)):
        directory = out / kind
        directory.mkdir()
        predictions = {person: {light: {label: guess(describe(recode(image)), model)
                                       for label, image in states.items()} for light, states in lights.items()}
                       for person, lights in people.items()}
        if json.loads(json.dumps(predictions)) != prior["encodings"][kind]["decisions"]["local"]:
            raise ValueError("Historical score replay differs")
        cases = {}
        for person, donor in permutations(GROUPS, 2):
            target = people[person]["cold"]["off"]
            grades = {"off": target} | {key: finish_spatial(target, render_style(target, op), STRENGTH)
                                         for key, op in ops.items()}
            area = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
            candidate, original = {}, {}
            for light, states in predictions[donor].items():
                candidate[light], original[light] = {}, {}
                for role, (label, values) in states.items():
                    same = np.array_equal(target, recode(people[donor][light][role]))
                    candidate[light][role] = target if same else mixture(grades, values)
                    original[light][role] = grades[label]
            reports = {}
            for arm, frames in (("hard", original), ("soft", candidate)):
                reports[arm] = {style: score(target, grades[style],
                                              {light: {"off": v["off"], "on": v[style]} for light, v in frames.items()}, area)
                                for style in ("film_a", "film_b")}
            name = f"{person}-from-{donor}"
            cases[name] = dict(scores=reports,
                hashes={f"{light}-{key}": save(directory / f"{name}-{light}-{key}.png", img)
                        for light, states in candidate.items() for key, img in states.items()},
                panel_sha=panel(directory / f"{name}-comparison.png", {"desired": grades} |
                    {f"soft {light}": frames for light, frames in candidate.items()} |
                    {f"hard {light}": frames for light, frames in original.items()}))
        suite[kind] = dict(weights={person: {light: {role: fractions(pair[1]).tolist() for role, pair in values.items()}
                                            for light, values in states.items()} for person, states in predictions.items()},
                           cases=cases)
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, suites=suite, script_sha=digest(script),
             protocol_sha=digest(script.with_name("gros-continuous-style-replay-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
