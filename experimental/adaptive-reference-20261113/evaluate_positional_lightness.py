"""Render and score the fully held-out capture families and separate warmth guard."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_arri_shared_look import metrics
from probe_cross_source_wall_lightening_workbench import DIRECTIONS, OUTDOOR_SHA, frozen_scenes, load
from probe_matched_wall_lightening_workbench import PARENT, scores
from probe_shared_local_warmth import rectangles
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel
from spatial_render import SpatialEngine, finish_spatial
from train_positional_lightness import ConditionedEngine, FAMILIES, PROTOCOL, STEPS


WARM_SHA = "0e040a54c3de7b6f3124664748cf592e7d3913705cc7c5610e074c081354c02c"
PORTRAITS = (("2", "warm"), ("4", "cold"))


def baseline(target, reference, engine):
    if np.array_equal(target, reference):
        return target.copy()
    raw = engine.prepare(target, reference).apply(target)
    return finish_spatial(target, raw, STRENGTH)


def save(path, rgb):
    Image.fromarray(rgb).save(path)
    return digest(path)


def areas_for(original, scene):
    areas = rectangles(original.shape, {**scene["boxes"], "extent": scene["extent"]})
    areas["outside"] = ~areas.pop("extent")
    return areas


def evaluate_lightness(scenes, family, render, path):
    results = {}
    left, right = FAMILIES[family]
    for target, donor in ((left, right), (right, left)):
        folder = path / target
        folder.mkdir(parents=True)
        chosen = scenes[target]
        anchor = chosen["anchor"]
        original = load(chosen, f"{anchor}-off")[1]
        authored = load(chosen, f"{anchor}-on")[1]
        answer = finish_spatial(original, authored, STRENGTH)
        areas = areas_for(original, chosen)
        own = render(original, original)
        same_image = dict(exact=bool(np.array_equal(original, own)), sha256=save(folder / "identity.png", own))
        pictures, hashes, cases = {}, {}, {}
        tiles = {"target": {"original": original, "answer": answer}}
        for name in DIRECTIONS[donor]:
            a, b = name.split("-to-")
            tags = dict(base=f"{a}-off", captured=f"{b}-off", edited=f"{a}-on")
            for tag in tags.values():
                if tag not in pictures:
                    reference = load(scenes[donor], tag)[1]
                    first = render(original, reference)
                    pictures[tag] = first
                    hashes[tag] = save(folder / f"{tag}.png", first)
            outputs = {role: pictures[tag] for role, tag in tags.items()}
            measured = scores(original, answer, outputs, areas)
            passed = same_image["exact"] and all(row["joint_passed"] for row in measured.values())
            cases[name] = dict(roles=tags, measures=measured, passed=passed)
            tiles[name] = outputs
        results[target] = dict(donor=donor, cases=cases, same_image=same_image, hashes=hashes,
                               board=panel(folder / "comparison.png", tiles))
    return results


def evaluate_warmth(record, render, path):
    results = {}
    for target, donor in (PORTRAITS, PORTRAITS[::-1]):
        who, level = target
        source, light = donor
        target_scene, donor_scene = record[who], record[source]
        original = load(target_scene, f"{level}-off")[1]
        expected = finish_spatial(original, load(target_scene, f"{level}-on")[1], STRENGTH)
        before = render(original, load(donor_scene, f"{light}-off")[1])
        after = render(original, load(donor_scene, f"{light}-on")[1])
        folder = path / "gros" / who
        folder.mkdir(parents=True)
        identity = render(original, original)
        same_image = dict(exact=bool(np.array_equal(identity, original)), sha256=save(folder / "identity.png", identity))
        rows = {name: metrics(original, expected, before, after, area)
                for name, area in areas_for(original, target_scene).items()}
        passed = (same_image["exact"] and rows["wall_left"]["passed"] and all(max(row["off_matches_original_mae"],
                  row["on_matches_authored_mae"], row["response_mae"]) <= 3 for row in rows.values()))
        pictures = {"original": original, "expected": expected, "off": before, "on": after}
        hashes = {name: save(folder / f"{name}.png", image) for name, image in pictures.items()}
        board = panel(folder / "comparison.png", {f"{who} from {source}": pictures})
        results[who] = dict(donor=source, measurements=rows, passed=passed, hashes=hashes, same_image=same_image, board=board)
    return results


def compare_arms(reports):
    directions = {name: {f"{target}/{direction}" for target, pair in result["lightness"].items()
                  for direction, info in pair["cases"].items() if info["passed"]}
                  for name, result in reports.items()}
    local = directions["local"]
    comparisons = {name: dict(added=sorted(local - directions[name]), lost=sorted(directions[name] - local))
                   for name in ("13", "pooled")}
    return dict(directions={name: sorted(value) for name, value in directions.items()},
                comparisons=comparisons,
                lightness_improved=all(item["added"] and not item["lost"] for item in comparisons.values()),
                warmth_preserved=all(row["passed"] for row in reports["local"]["warmth"].values()))


def run(base, training, out):
    torch.set_num_threads(4)
    original = checked_json(base / "shared-local-warmth-v2/results.json", WARM_SHA)["results"]["gros"]
    for who, rec in original.items():
        rec["folder"] = base / "shared-local-warmth-v2/gros" / who
    source = training / "training.json"
    book = json.loads(source.read_bytes())
    if (book["preflight"] or book["protocol"] != PROTOCOL or book["indoor"] != PARENT
            or book["outdoor"] != OUTDOOR_SHA or book["strength"] != STRENGTH
            or book["script"] != digest(Path(__file__).with_name("train_positional_lightness.py"))):
        raise ValueError("Training receipts do not identify this frozen experiment")
    scenes = frozen_scenes(base)
    out.mkdir()
    current = SpatialEngine("multiref-v1")
    result = {}
    for train, test in (tuple(FAMILIES), tuple(reversed(FAMILIES))):
        if set(FAMILIES[train]) & set(FAMILIES[test]):
            raise ValueError("A held-out scene also appears in fitting")
        fold = {}
        for arm in ("13", "local", "pooled"):
            if arm == "13":
                def render(target, reference):
                    return baseline(target, reference, current)
            else:
                weight = training / train / f"{arm}.pth"
                arm_record = book["folds"][train][arm]
                if arm_record["steps"] != STEPS or arm_record["parent"] != CHECKPOINT or digest(weight) != arm_record["head_sha256"]:
                    raise ValueError("Trained head weights do not match the fixed run")
                engine = ConditionedEngine(arm == "local")
                engine.head.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
                render = engine.render
            path = out / test / arm
            fold[arm] = dict(lightness=evaluate_lightness(scenes, test, render, path),
                             warmth=evaluate_warmth(original, render, path))
        result[test] = dict(trained_on=train, arms=fold, outcome=compare_arms(fold))
        print(test, result[test]["outcome"], flush=True)
    json_to(out/"results.json", dict(training=digest(source), warmth=WARM_SHA, protocol=PROTOCOL,
                                    script=digest(Path(__file__)), results=result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("base", "training", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
