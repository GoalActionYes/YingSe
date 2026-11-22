"""Replay known local editor differences across people using the unchanged spatial renderer."""
from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from capacity_audit import block_split
from probe_gros_grid_expression import rendering_features
from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, areas_and_weight, score_outputs
from probe_gros_local_grid_capacity import GRID, axis_weights, evaluate, fit, prepare, spatial_features, support
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


PARENT = "0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7"
CAPACITY = "252711ece1da355cf50335ad4062f4b52294b0cc7edad38772023cb7e0b2a12e"


def landmarks(person):
    left, right = [np.asarray(person[key], dtype=float).reshape(2, 2).mean(axis=0)
                   for key in ("cheek_left", "cheek_right")]
    dimensions = np.array([384., 256.])
    middle = (left + right) / dimensions - 1
    return middle, float(np.linalg.norm(right - left))


@torch.inference_mode()
def transport(field, source, target):
    source_center, source_span = source
    dest_center, dest_span = target
    if source_span == dest_span and np.array_equal(source_center, dest_center):
        return field
    lanes = (2 * torch.arange(GRID, device=field.device, dtype=torch.float32) + 1) / GRID - 1
    y, x = torch.meshgrid(lanes, lanes, indexing="ij")
    coords = torch.stack((x, y), dim=-1)
    src = torch.as_tensor(source_center, device=field.device, dtype=torch.float32)
    dest = torch.as_tensor(dest_center, device=field.device, dtype=torch.float32)
    grid = ((coords - dest) * (source_span / dest_span) + src)[None]
    moved = F.grid_sample(field.flatten(1, 2), grid, align_corners=False, padding_mode="border")
    return moved.reshape_as(field)


def verify_geometry(device):
    ramp = torch.arange(GRID, dtype=torch.float32, device=device)[None, None, None, None]
    ramp = ramp.expand(1, 1, 1, GRID, GRID).contiguous()
    centered = np.array([0., 0.]), 40.
    shifted = np.array([2. / GRID, 0.]), 40.
    identity = transport(ramp, centered, centered)
    moved = transport(ramp, centered, shifted)
    torch.testing.assert_close(identity, ramp, atol=0, rtol=0)
    torch.testing.assert_close(moved[..., 1:], ramp[..., :-1], atol=1e-6, rtol=0)


def replay(engine, target, fields):
    off, on = [PreparedSpatial(engine, grid, {}).float_output(target) for grid in fields]
    delta = on - off
    wanted = target.astype(np.float32) + delta * 255
    clip = float(np.any((wanted < 0) | (wanted > 255), axis=-1).mean())
    picture = np.rint(np.clip(wanted, 0, 255)).astype(np.uint8)
    return finish_spatial(target, picture, STRENGTH), clip


def fit_pair(engine, source, edited, regions):
    h, w = source.shape[:2]
    affine, depth, after = rendering_features(engine, source, edited)
    _, _, before = rendering_features(engine, source, source)
    x_nodes = support(axis_weights(w, engine.device))
    y_nodes = support(axis_weights(h, engine.device))
    train = block_split(h, w, 16).reshape(h, w)
    yy, xx = np.ogrid[:h, :w]
    rows = np.flatnonzero(train & (yy % 3 == 0) & (xx % 3 == 0))
    samples = spatial_features(affine, depth, x_nodes, y_nodes, rows)
    fields, audits = [], {}
    for name, answer in (("off", before), ("on", after)):
        estimates, audit = fit(affine, depth, answer, rows, samples)
        fields.append(prepare(engine, estimates["spatial"]).grid)
        audits[name] = audit
    control, clip = replay(engine, source, fields)
    expected = finish_spatial(source, edited, STRENGTH)
    held = evaluate(source, expected, control, {name: mask & ~train for name, mask in regions.items()})
    return fields, {"held": held, "clipped": clip, "fits": audits}, control


def run(project, output):
    if scipy.__version__ != "1.18.1":
        raise ValueError("Use the same frozen sparse solver as the capacity check")
    root = project / "gros-local-editor-channel-v2"
    parent = checked_json(root / "results.json", PARENT)
    checked_json(project / "gros-local-grid-capacity-v1/results.json", CAPACITY)
    script = Path(__file__)
    coords = frozen_json(script.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed checkpoint")
    output.mkdir()
    verify_geometry(engine.device)
    subjects = parent["qualified"]
    source_fields, geometry, zones, audits, originals = {}, {}, {}, {}, {}
    for person in subjects:
        geometry[person] = landmarks(coords[person])
        zones[person], alpha = areas_and_weight(coords[person], coords["screen"])
        zones[person]["outside"] = alpha == 0
        for light, receipt in parent["intake"][person].items():
            src, edit = [read_frame(root / person / light / f"{role}.png", receipt["hashes"][role])
                         for role in ("off", "on")]
            key = person, light
            originals[key] = src
            source_fields[key], audits[f"{person}-{light}"], image = fit_pair(engine, src, edit, zones[person])
            audits[f"{person}-{light}"]["render_sha"] = save(output / f"{person}-{light}-self.png", image)
            print(f"Fitted reference {person}-{light}", flush=True)

    cases = {}
    for target, donor in permutations(subjects, 2):
        original = originals[target, "cold"]
        slug = f"{target}-from-{donor}"
        frames, clipping, hashes = {}, {}, {}
        for arm in ("image_position", "face_alignment"):
            frames[arm], clipping[arm], hashes[arm] = {}, {}, {}
            for light in ("cold", "warm"):
                direct = source_fields[donor, light]
                grids = direct if arm == "image_position" else [transport(grid, geometry[donor], geometry[target])
                                                               for grid in direct]
                image, clipped = replay(engine, original, grids)
                frames[arm][light] = {"off": original, "on": image}
                clipping[arm][light] = clipped
                hashes[arm][light] = save(output / f"{slug}-{arm}-{light}.png", image)
        expected = read_frame(root / f"{slug}-expected.png", parent["cases"][slug]["wanted"])
        scores = score_outputs(original, expected, frames, zones[target])
        for arm, results in scores.items():
            outside = results["outside"]
            outside["passed"] = (max(outside["lamp_spill"].values()) <= 3 and all(
                item["on_matches_authored_mae"] <= 3 for item in outside["response"].values()))
        compare = {"target": {"original": original, "expected": expected}}
        compare.update({name: {light: states["on"] for light, states in view.items()} for name, view in frames.items()})
        cases[slug] = dict(scores=scores, clipping=clipping, hashes=hashes,
                           complete={name: all(row["passed"] for row in values.values()) for name, values in scores.items()},
                           panel=panel(output / f"{slug}-comparison.png", compare))
    json_to(output / "results.json", dict(parent=PARENT, capacity=CAPACITY, checkpoint=CHECKPOINT, strength=STRENGTH,
            native_grid=GRID, geometry_control=True, reference_held=audits, cases=cases, script_sha=digest(script),
            protocol_sha=digest(script.with_name("gros-local-grid-transfer-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
