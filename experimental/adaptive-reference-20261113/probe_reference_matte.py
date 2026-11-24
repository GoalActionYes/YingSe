"""Measure the effect of removing exact framing from a cross-scene reference."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch

from probe_cross_source_wall_lightening_workbench import DIRECTIONS, TARGET_DONOR, frozen_scenes, load
from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score as portrait_score
from probe_matched_wall_lightening_workbench import scores as wall_scores
from probe_mid_matched_wall_cast import box
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


WALL_SHA = "d9caefeff6a92f7bbde83347aa687dc8959e4f51f75421792a6f0bbc6e33da3a"
PORTRAIT_SHA = "4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260"
PORTRAIT_REGIONS_SHA = "d69083602c80e65a859dd90531861cd283dffe99de90fb878c981d58315235c5"
FACD_SHA = "c85b72723bf4651fd51942790022f6e541b0e7af358b33aa8a4c8e29773d71f7"
AXES = (0, 1)
ARMS = ("plain", "horizontal", "vertical", "cropped-horizontal", "cropped-vertical")


def matte_bounds(rgb):
    black = np.all(rgb == 0, axis=2)
    accepted = []
    for axis in AXES:
        lines = np.flatnonzero(~black.all(axis=1-axis))
        if not len(lines):
            return None
        leading = int(lines[0])
        stop = int(lines[-1]) + 1
        trailing = rgb.shape[axis] - stop
        if (min(leading, trailing) >= 4 and abs(leading - trailing) <= 1
                and 5 * max(leading, trailing) <= rgb.shape[axis]):
            accepted.append((axis, leading, stop))
    return accepted[0] if len(accepted) == 1 else None


def picture(rgb):
    bounds = matte_bounds(rgb)
    if bounds is None:
        return rgb
    axis, start, stop = bounds
    return rgb[start:stop] if axis == 0 else rgb[:, start:stop]


def pad(rgb, axis):
    border = rgb.shape[axis] // 8
    values = tuple((border, border) if i == axis else (0, 0) for i in range(3))
    return np.pad(rgb, values)


def mae(before, after, area):
    return float(np.abs(before[area].astype(np.int16) - after[area]).mean())


def grade(engine, target, reference):
    if np.array_equal(target, reference):
        return target.copy()
    return finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)


def render_inputs(engine, target, references, plain_paths, out):
    out.mkdir(parents=True)
    images = {arm: {} for arm in ARMS}
    sources = {}
    hashes = {arm: {} for arm in ARMS}
    border_sha = {}
    for name, rgb in references.items():
        padded = {arm: pad(rgb, axis) for axis, arm in zip(AXES, ARMS[1:3], strict=True)}
        inputs = {"plain": rgb, **padded}
        for arm, padded_rgb in padded.items():
            border_sha[f"{name}-{arm}"] = save(out / f"{name}-reference-{arm}.png", padded_rgb)
            cleaned = picture(padded_rgb)
            np.testing.assert_array_equal(rgb, cleaned)
            inputs["cropped-" + arm] = cleaned
        for arm, guide in inputs.items():
            image = grade(engine, target, guide)
            images[arm][name] = image
            hashes[arm][name] = save(out / f"{name}-{arm}.png", image)
            if arm.startswith("cropped-"):
                np.testing.assert_array_equal(image, images["plain"][name])
        frozen_file, frozen_hash = plain_paths[name]
        np.testing.assert_array_equal(images["plain"][name], checked_rgb(frozen_file, frozen_hash))
        sources[name] = {key: inputs[key] for key in ARMS[:3]}
    return images, sources, hashes, border_sha


def interventions(outputs, target, wanted, areas, roles):
    rows = {}
    for arm in ARMS[1:3]:
        clean = outputs["cropped-" + arm]
        changed = outputs[arm]
        rows[arm] = {}
        for name, area in areas.items():
            parts = {}
            for role, tag in roles.items():
                expected = wanted if role == "on" else target
                padded_error = mae(changed[tag], expected, area)
                recovered_error = mae(clean[tag], expected, area)
                parts[role] = dict(padded_error=padded_error, recovered_error=recovered_error,
                                   reduction=padded_error-recovered_error,
                                   drift=mae(changed[tag], clean[tag], area))
            rows[arm][name] = parts
    return rows


def walls(engine, base, out):
    archive = base / "cross-source-wall-lightening-workbench-v2"
    previous = checked_json(archive / "results.json", WALL_SHA)["reports"]["spatial13"]
    scenes = frozen_scenes(base)
    summary = {}
    for target, donor in TARGET_DONOR:
        scene = scenes[target]
        anchor = scene["anchor"]
        _, before = load(scene, f"{anchor}-off")
        _, authored = load(scene, f"{anchor}-on")
        expected = finish_spatial(before, authored, STRENGTH)
        areas = {name: box(before, bounds) for name, bounds in scene["boxes"].items()}
        x0, y0, x1, y1 = scene["extent"]
        y, x = np.ogrid[:before.shape[0], :before.shape[1]]
        areas["outside"] = ~((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
        roles = {}
        for direction in DIRECTIONS[donor]:
            a, b = direction.split("-to-")
            roles[direction] = dict(base=f"{a}-off", captured=f"{b}-off", edited=f"{a}-on")
        refs = {tag: load(scenes[donor], tag)[1]
                for tag in sorted({tag for batch in roles.values() for tag in batch.values()})}
        paths = {tag: (archive / "spatial13" / target / f"{tag}.png", previous[target]["outputs"][tag])
                 for tag in refs}
        here = out / target
        results, reference_images, hashes, borders = render_inputs(engine, before, refs, paths, here)
        details = {}
        for direction, labels in roles.items():
            comparison = {arm: wall_scores(before, expected,
                          {role: states[tag] for role, tag in labels.items()}, areas)
                          for arm, states in results.items()}
            impact = interventions(results, before, expected, areas,
                                   dict(off=labels["base"], captured=labels["captured"], on=labels["edited"]))
            details[direction] = dict(metrics=comparison, impact=impact)
        tag = next(iter(roles.values()))["edited"]
        board = panel(here / "comparison.png", {
            "target": dict(original=before, intended=expected),
            "reference": reference_images[tag],
            "output": {arm: states[tag] for arm, states in results.items()},
        })
        summary[target] = dict(donor=donor, details=details, output=hashes, borders=borders, board=board)
        print("wall", target, len(details), flush=True)
    return summary


def portraits(engine, base, out):
    archive = base / "gros-film-b-cross-v2"
    previous = checked_json(archive / "results.json", PORTRAIT_SHA)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != PORTRAIT_REGIONS_SHA:
        raise ValueError("Changed portrait measurement regions")
    registered = json.loads(ledger.read_bytes())

    def source(person, light, state):
        name = f"{person}-{light}"
        return checked_rgb(archive / f"{name}-{state}.png", previous["inputs"][name]["hashes"][state])

    summary = {}
    for target, donor in permutations(GROUPS, 2):
        pair = f"{target}-from-{donor}"
        before = source(target, "cold", "off")
        expected = finish_spatial(before, source(target, "cold", "on"), STRENGTH)
        area = masks(registered[target]["cold"], registered["screen"], before.shape[:2])
        options = {f"{light}-{state}": source(donor, light, state) for light in LIGHTS for state in ("off", "on")}
        paths = {tag: (archive / f"{pair}-13-{tag}.png", previous["cases"][pair]["hashes"][f"13-{tag}"])
                 for tag in options}
        here = out / pair
        results, reference_images, hashes, borders = render_inputs(engine, before, options, paths, here)
        comparison = {arm: portrait_score(before, expected, {light: {state: states[f"{light}-{state}"]
                       for state in ("off", "on")} for light in LIGHTS}, area) for arm, states in results.items()}
        impact = {light: interventions(results, before, expected, area, dict(off=f"{light}-off", on=f"{light}-on"))
                  for light in LIGHTS}
        tag = "warm-on"
        board = panel(here / "comparison.png", {
            "target": dict(original=before, intended=expected),
            "reference": reference_images[tag],
            "output": {arm: states[tag] for arm, states in results.items()},
        })
        summary[pair] = dict(metrics=comparison, impact=impact, output=hashes, borders=borders, board=board)
        print("portrait", pair, flush=True)
    return summary


def negative_controls(engine, base):
    receipt = Path(__file__).resolve().parents[2] / "docs/research/2026-11-18-facd-range-pilot-evidence.json"
    prior = checked_json(receipt, FACD_SHA)
    source = base / "facd-range-media-v1"
    findings = {}
    for group, states in prior["cases"].items():
        for state, evidence in states.items():
            path = source / group / f"{state}.jpg"
            if digest(path) != evidence["sha256"]:
                raise ValueError(f"Changed negative fixture: {group}/{state}")
            frame = decode_image(path)
            bounds = matte_bounds(frame)
            findings[f"{group}/{state}"] = dict(bounds=bounds, exact=np.array_equal(frame, picture(frame)))
    target = np.arange(80*100*3, dtype=np.uint32).astype(np.uint8).reshape(80, 100, 3)
    fixtures = {"unframed": target, "empty": np.zeros_like(target),
                "one-sided": np.pad(target, ((12, 0), (0, 0), (0, 0))),
                "asymmetric": np.pad(target, ((12, 8), (0, 0), (0, 0))),
                "near-black": np.pad(target, ((12, 12), (0, 0), (0, 0)), constant_values=1),
                "black-frame": np.pad(target, ((8, 8), (8, 8), (0, 0))),
                "partial": np.pad(target, ((12, 12), (0, 0), (0, 0)))}
    fixtures["partial"][:, 2] = 16
    synthetic = {key: bool(np.array_equal(value, picture(value))) for key, value in fixtures.items()}
    identity = all(np.array_equal(grade(engine, target, item), target)
                   for item in (target, picture(pad(target, 0)), picture(pad(target, 1))))
    return dict(facd=findings, synthetic=synthetic, identity=identity)


def impact_records(walls, portrait):
    for target, entry in walls.items():
        for condition, row in entry["details"].items():
            for format_name, areas in row["impact"].items():
                for region, roles in areas.items():
                    for role, item in roles.items():
                        yield ("wall", target, condition, format_name, region, role), item
    for pair, entry in portrait.items():
        for condition, group in entry["impact"].items():
            for format_name, areas in group.items():
                for region, roles in areas.items():
                    for role, item in roles.items():
                        yield ("portrait", pair, condition, format_name, region, role), item


def run(base, out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or STRENGTH != .8:
        raise ValueError("Changed frozen engine or strength")
    out.mkdir()
    controls = negative_controls(engine, base)
    changes = walls(engine, base, out / "walls")
    people = portraits(engine, base, out / "portraits")
    impacts = list(impact_records(changes, people))
    potential = {family: [dict(case=label, **item) for label, item in impacts
                          if label[0] == family and label[4] not in ("whole", "outside") and item["reduction"] >= 3]
                 for family in ("wall", "portrait")}
    regressions = [dict(case=label, **item) for label, item in impacts if item["reduction"] < -3]
    no_crop = all(row["exact"] for row in controls["facd"].values()) and all(controls["synthetic"].values())
    passed = no_crop and controls["identity"] and all(potential.values()) and not regressions
    code = Path(__file__)
    json_to(out / "results.json", dict(checkpoint=CHECKPOINT, strength=STRENGTH, parent_wall=WALL_SHA,
            parent_portrait=PORTRAIT_SHA, script=digest(code), protocol=digest(code.with_name("reference-matte-protocol.md")),
            controls=controls, walls=changes, portrait=people, gains=potential, regressions=regressions, passed=passed))
    print("regional gains >=3", {key: len(value) for key, value in potential.items()},
          "regressions >3", len(regressions), "controls", no_crop, "proceed", passed)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
