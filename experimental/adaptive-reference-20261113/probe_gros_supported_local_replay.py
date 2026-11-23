"""Audit spatial and observed-color support on frozen local grade replay images."""
from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import ConvexHull, QhullError
import torch
import torch.nn.functional as F

from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, areas_and_weight, score_outputs
from probe_gros_local_grid_transfer import landmarks
from probe_physical_blue_global_fit import linear
from probe_target_mirror_chroma import checked_json, digest, json_to, panel, save


PARENT = "0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7"
REPLAY = "e673eb578249c7b6a73d085c82dfb5c2dedbe1057f69ea2c482793567ec3de67"
ARMS = ("position", "position_and_color")
LIGHTS = ("cold", "warm")


def warp(mask, donor, target):
    source_at, source_span = donor
    target_at, target_span = target
    if source_span == target_span and np.array_equal(source_at, target_at):
        return mask.astype(np.float32)
    h, w = mask.shape
    xx = (2 * torch.arange(w, dtype=torch.float32) + 1) / w - 1
    yy = (2 * torch.arange(h, dtype=torch.float32) + 1) / h - 1
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((x, y), dim=-1)
    source = torch.as_tensor(source_at, dtype=torch.float32)
    dest = torch.as_tensor(target_at, dtype=torch.float32)
    coordinates = ((grid - dest) * (source_span / target_span) + source)[None]
    plane = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    return F.grid_sample(plane, coordinates, align_corners=False, padding_mode="zeros")[0, 0].numpy()


def hull(points):
    unique = np.unique(points, axis=0)
    if len(unique) < 4:
        return None
    try:
        return ConvexHull(unique).equations
    except QhullError:
        return None


def membership(points, bounds):
    valid = np.zeros(len(points), dtype=bool)
    if bounds is None:
        return valid
    planes, offset = bounds[:, :3], bounds[:, 3]
    for start in range(0, len(points), 4096):
        block = slice(start, start + 4096)
        valid[block] = ((points[block] @ planes.T) + offset <= 1e-10).all(axis=1)
    return valid


def apply_weight(original, graded, weight):
    delta = graded.astype(np.int16) - original
    return np.rint(original + delta * weight[..., None]).clip(0, 255).astype(np.uint8)


def controls():
    simple = np.eye(4, 3)
    bounds = hull(simple)
    assert bounds is not None
    queries = np.array([[.25, .25, .25], [0., 0., 0.], [.5, .5, .5], [-.01, .2, .2]])
    np.testing.assert_array_equal(membership(queries, bounds), [True, True, False, False])
    own = np.array([0., 0.]), 1.
    shape = np.eye(7, dtype=bool)
    weight = warp(shape, own, own)
    np.testing.assert_array_equal(weight, shape)
    original = np.arange(147, dtype=np.uint8).reshape(7, 7, 3)
    np.testing.assert_array_equal(apply_weight(original, original, weight), original)
    np.testing.assert_array_equal(apply_weight(original, 255-original, np.zeros_like(weight)), original)
    return {"tetrahedron": True, "same_geometry": True, "zero_effect": True, "zero_support": True}


def score_arms(original, expected, frames, areas):
    scores = score_outputs(original, expected, frames, areas)
    for rows in scores.values():
        outside = rows["outside"]
        outside["passed"] = (max(outside["lamp_spill"].values()) <= 3 and all(
            row["on_matches_authored_mae"] <= 3 for row in outside["response"].values()))
    return scores


def run(base, output):
    if scipy.__version__ != "1.18.1":
        raise ValueError("Use the frozen SciPy version")
    torch.set_num_threads(4)
    gates = controls()
    source = base / "gros-local-editor-channel-v2"
    replay = base / "gros-local-grid-transfer-v1"
    original = checked_json(source / "results.json", PARENT)
    previous = checked_json(replay / "results.json", REPLAY)
    script = Path(__file__)
    coords = frozen_json(script.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    output.mkdir()
    records, photos, supports, regions, locations = {}, {}, {}, {}, {}
    for name in original["qualified"]:
        regions[name], op = areas_and_weight(coords[name], coords["screen"])
        regions[name]["outside"] = op == 0
        locations[name] = landmarks(coords[name])
        for light in LIGHTS:
            frame = original["intake"][name][light]
            off, on = [read_frame(source / name / light / f"{state}.png", frame["hashes"][state])
                       for state in ("off", "on")]
            photos[name, light] = off
            edited = np.abs(on.astype(np.int16) - off).mean(axis=-1) >= 1
            bounds = hull(linear(np.unique(off[edited], axis=0)))
            supports[name, light] = (edited, bounds)
            records[f"{name}-{light}"] = dict(edited_pixels=int(edited.sum()),
                                                hull_planes=len(bounds) if bounds is not None else None)
    cases = {}
    for subject, donor in permutations(original["qualified"], 2):
        slug = f"{subject}-from-{donor}"
        parent = previous["cases"][slug]
        target = photos[subject, "cold"]
        target_linear = linear(target)
        frames, hashes = ({arm: {} for arm in ARMS} for _ in range(2))
        evidence, inputs = {}, {}
        for light in LIGHTS:
            arm_file = f"{slug}-face_alignment-{light}.png"
            known = read_frame(replay / arm_file, parent["hashes"]["face_alignment"][light])
            inputs[light] = known
            mask, bounds = supports[donor, light]
            position = warp(mask, locations[donor], locations[subject])
            spatial = position > 0
            admitted = np.zeros(position.shape, dtype=bool)
            admitted[spatial] = membership(target_linear[spatial], bounds)
            weights = dict(position=position, position_and_color=position * admitted)
            for arm, weight in weights.items():
                graded = apply_weight(target, known, weight)
                np.testing.assert_array_equal(graded[weight == 0], target[weight == 0])
                frames[arm][light] = {"off": target, "on": graded}
                hashes[arm][light] = save(output / f"{slug}-{arm}-{light}.png", graded)
            evidence[light] = {zone: {arm: float(weight[area].mean()) for arm, weight in weights.items()}
                               for zone, area in regions[subject].items()}
        expected = read_frame(source / f"{slug}-expected.png", original["cases"][slug]["wanted"])
        scores = score_arms(target, expected, frames, regions[subject])
        images = {"target": {"original": target, "expected": expected}, "prior aligned": inputs}
        images.update({name: {light: states["on"] for light, states in trace.items()} for name, trace in frames.items()})
        cases[slug] = dict(scores=scores, coverage=evidence, hashes=hashes,
                           complete={arm: all(r["passed"] for r in rows.values()) for arm, rows in scores.items()},
                           panel=panel(output / f"{slug}-comparison.png", images))
        print(slug, cases[slug]["complete"], flush=True)
    json_to(output / "results.json", dict(parent=PARENT, replay=REPLAY, scipy=scipy.__version__, controls=gates,
            reference_support=records, cases=cases, script_sha=digest(script),
            protocol_sha=digest(script.with_name("gros-supported-local-replay-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
