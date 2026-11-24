"""Exercise both workbenches on the frozen MIT matched wall-warmth reference."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_cross_source_wall_lightening_workbench import STABLE_SHA, load
from probe_gros_local_editor_channel import actual_png, request, serving
from probe_matched_wall_lightening_workbench import scores
from probe_shared_local_warmth import rectangles
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


PROTOCOL = "0a2df63a1da7a208b5c19917aa90675123a92c2a70ee6ec71d8f191f1720bf76"
SOURCE = "0e040a54c3de7b6f3124664748cf592e7d3913705cc7c5610e074c081354c02c"
DONOR = "14n_copyroom1"
REFERENCE = {"base": "8-off", "captured": "0-off", "edited": "8-on"}
TARGETS = (
    ("mit", "everett_living2", "0"),
    ("wildrelight", "wall", "time0"),
    ("gros", "2", "cold"),
    ("gros", "3", "cold"),
    ("gros", "4", "cold"),
)


def target_areas(original, scene, portrait):
    areas = rectangles(original.shape, {**scene["boxes"], "extent": scene["extent"]})
    areas["outside"] = ~areas.pop("extent")
    if portrait:
        areas["cheek_right"] = areas.pop("wall_left")
        areas["cheek_left"] = areas.pop("other_cheek")
    if not areas["outside"].any():
        raise ValueError("No outside reference region")
    return areas


def compare_target(server, scenes, reference_ids, reference_images, family, name, level, out):
    scene = scenes[family][name]
    path, original = load(scene, f"{level}-off")
    if not scene["frames"][level]["metrics"]["admitted"]:
        raise ValueError(f"Unqualified target source: {family}/{name}/{level}")
    portrait = family == "gros"
    expected = original if portrait else finish_spatial(original, load(scene, f"{level}-on")[1], STRENGTH)
    areas = target_areas(original, scene, portrait)
    if not portrait and np.array_equal(original[areas["wall_left"]], expected[areas["wall_left"]]):
        raise ValueError("No edited-wall signal in this target")
    out.mkdir(parents=True)
    target_id = request(server, "/api/media", path.read_bytes(), f"{name}-{path.name}")["id"]
    identity, identity_sha = actual_png(server, target_id, target_id, out / "identical.png")
    exact = bool(np.array_equal(identity, original))
    exports, hashes = {}, {}
    for role, reference_id in reference_ids.items():
        exports[role], hashes[role] = actual_png(server, target_id, reference_id, out / f"{role}.png")
        if exports[role].shape != original.shape:
            raise ValueError(f"Unexpected export dimensions: {family}/{name}/{role}")

    measured = scores(original, expected, exports, areas)
    if portrait:
        for region, row in measured.items():
            captured = exports["captured"][areas[region]].astype(np.int16)
            captured_original = float(np.abs(captured - original[areas[region]]).mean())
            row["captured_matches_original_mae"] = captured_original
            row["joint_passed"] = row["joint_passed"] and captured_original <= 3
    passed = exact and all(row["joint_passed"] for row in measured.values())
    board = panel(out / "comparison.png", {
        "target": {"original": original, "expected": expected},
        "reference": reference_images,
        "software export": exports,
    })
    return dict(family=family, target=f"{name}/{level}-off", wall_target=not portrait, passed=passed,
                identical=dict(exact=exact, sha256=identity_sha), outputs=hashes,
                measured=measured, board_sha256=board)


def run(base, out):
    torch.set_num_threads(4)
    script = Path(__file__)
    if (digest(script.with_name("matched-wall-warmth-workbench-protocol.md")) != PROTOCOL
            or algorithm_fingerprint() != STABLE_SHA or STRENGTH != .8):
        raise ValueError("Changed comparison protocol or workbench")
    input_dir = base / "shared-local-warmth-v2"
    scenes = checked_json(input_dir / "results.json", SOURCE)["results"]
    for family, rows in scenes.items():
        for name, scene in rows.items():
            scene["folder"] = input_dir / family / name
    donor = scenes["mit"][DONOR]
    matched = donor["directions"]["8-to-0"]
    if not matched["passed"]:
        raise ValueError("Fixed source direction did not qualify")
    sources = {role: load(donor, tag) for role, tag in REFERENCE.items()}
    out.mkdir()
    result = {}
    for kind, factory, options in (
            ("stable", make_server, {}),
            ("spatial13", make_spatial_workbench, {"enhanced": True})):
        root = out / kind
        with serving(factory(root / "session", **options)) as server:
            if kind == "spatial13" and server.store.engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Workstation loaded the wrong checkpoint")
            ids = {role: request(server, "/api/media", frame.read_bytes(), f"{DONOR}-{frame.name}")["id"]
                   for role, (frame, _) in sources.items()}
            preview = {role: rgb for role, (_, rgb) in sources.items()}
            reports = {}
            for family, name, level in TARGETS:
                target = f"{family}-{name}"
                row = compare_target(server, scenes, ids, preview, family, name, level, root / target)
                reports[target] = row
                print(kind, target, row["passed"], flush=True)
            result[kind] = reports
    json_to(out / "results.json", dict(source=SOURCE, donor=DONOR, direction=matched, reference=REFERENCE,
                                       script=digest(script), protocol=PROTOCOL, strength=STRENGTH,
                                       stable=STABLE_SHA, checkpoint=CHECKPOINT, workbenches=result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
