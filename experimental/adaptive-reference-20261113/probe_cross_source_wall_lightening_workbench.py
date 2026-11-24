"""Compare actual workbench exports for matched capture and authored references."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_gros_local_editor_channel import actual_png, request, serving
from probe_matched_wall_lightening_workbench import PARENT, SOURCE, scores
from probe_mid_matched_wall_cast import box
from probe_playraw_flash_room import read
from probe_shared_authored_tone_admission import ROOM_SHA, ROOMS
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


OUTDOOR = "wildrelight-shared-tone-v1"
OUTDOOR_SHA = "915122b02c31e69b79f070130b7b9ce11ebe268d81d7fd6d0d0062077e1a730b"
STABLE_SHA = "c1d65198cfa49ea4655d4eceb03421f6cb6c688131576cee4ce9c36f77aa2ee1"
PROTOCOL_SHA = "b885352ce191daa71cf7d45fcb74ad5f8d694e277dba877d5ba1f70e57443238"
DIRECTIONS = {
    "wall": ("time5-to-time1", "time5-to-time2", "time2-to-time0"),
    "pillar": ("time5-to-time0", "time5-to-time1", "time1-to-time2"),
    ROOMS[0]: ("4-to-20", "12-to-8", "14-to-20", "16-to-20"),
    ROOMS[1]: ("4-to-8", "8-to-0", "12-to-0", "14-to-8", "16-to-4",
               "16-to-14", "20-to-4", "20-to-14"),
}
TARGET_DONOR = ((ROOMS[0], "wall"), (ROOMS[1], "pillar"),
                ("wall", ROOMS[0]), ("pillar", ROOMS[1]))


def frozen_scenes(base):
    indoors = checked_json(base / SOURCE / "results.json", PARENT)["rooms"]
    outdoors = checked_json(base / OUTDOOR / "results.json", OUTDOOR_SHA)["scenes"]
    ledger = checked_json(base / "mid-gray-anchor-v1.json", ROOM_SHA)["scenes"]
    scenes = {}
    for name, record in indoors.items():
        scenes[name] = dict(folder=base / SOURCE / name, frames=record["renders"],
                            comparisons=record["comparisons"], anchor="0",
                            boxes=ledger[name]["boxes"], extent=record["extent"])
    for name, record in outdoors.items():
        bounds = record["rectangles"]
        scenes[name] = dict(folder=base / OUTDOOR / name, frames=record["cases"],
                            comparisons=record["comparison"], anchor="time0",
                            boxes={"wall_left": bounds["edit"], "protect": bounds["protect"]},
                            extent=record["extent"])
    for name, scene in scenes.items():
        passed = {direction for direction, row in scene["comparisons"].items() if row["passed"]}
        if passed != set(DIRECTIONS[name]) or not scene["frames"][scene["anchor"]]["metrics"]["admitted"]:
            raise ValueError(f"Changed development cohort: {name}")
    return scenes


def load(scene, tag):
    level, state = tag.split("-")
    path = scene["folder"] / f"{tag}.png"
    rgb, _, identity = read(path, False)
    frozen = scene["frames"][level]["files"][state]
    if any(identity[key] != frozen[key] for key in ("pixels_sha256", "icc_sha256")):
        raise ValueError(f"Changed prepared picture: {path}")
    return path, rgb


def export_cases(server, scenes, target, donor, out):
    out.mkdir()
    scene = scenes[target]
    anchor = scene["anchor"]
    path, before = load(scene, f"{anchor}-off")
    _, after = load(scene, f"{anchor}-on")
    expected = finish_spatial(before, after, STRENGTH)
    target_id = request(server, "/api/media", path.read_bytes(), f"{target}-{path.name}")["id"]
    own, own_sha = actual_png(server, target_id, target_id, out / "identical.png")
    identical = bool(np.array_equal(before, own))

    areas = {name: box(before, bounds) for name, bounds in scene["boxes"].items()}
    x0, y0, x1, y1 = scene["extent"]
    y, x = np.ogrid[:before.shape[0], :before.shape[1]]
    areas["outside"] = ~((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    if any(not area.any() for area in areas.values()):
        raise ValueError(f"Missing measurement area: {target}")

    outputs, hashes, cases = {}, {}, {}
    tiles = {"target": {"original": before, "expected": expected}}
    for direction in DIRECTIONS[donor]:
        a, b = direction.split("-to-")
        roles = dict(base=f"{a}-off", captured=f"{b}-off", edited=f"{a}-on")
        reference_tiles = {}
        for role, tag in roles.items():
            if tag not in outputs:
                reference_path, reference = load(scenes[donor], tag)
                reference_id = request(server, "/api/media", reference_path.read_bytes(),
                                       f"{donor}-{reference_path.name}")["id"]
                picture, hashes[tag] = actual_png(server, target_id, reference_id, out / f"{tag}.png")
                if picture.shape != before.shape:
                    raise ValueError(f"Changed output dimensions: {target} from {donor}, {tag}")
                outputs[tag] = (picture, reference)
            reference_tiles[role] = outputs[tag][1]
        current = {role: outputs[tag][0] for role, tag in roles.items()}
        measurements = scores(before, expected, current, areas)
        cases[direction] = dict(reference_states=roles, scores=measurements,
                                passed=identical and all(row["joint_passed"] for row in measurements.values()))
        tiles[f"reference {direction}"] = reference_tiles
        tiles[f"export {direction}"] = current
    board = panel(out / "comparison.png", tiles)
    return dict(donor=donor, target_input=f"{anchor}-off", same_image=dict(exact=identical, sha256=own_sha),
                outputs=hashes, cases=cases, board=board)


def run(base, out):
    torch.set_num_threads(4)
    code = Path(__file__)
    protocol = code.with_name("cross-source-wall-lightening-workbench-protocol.md")
    if digest(protocol) != PROTOCOL_SHA or algorithm_fingerprint() != STABLE_SHA or STRENGTH != .8:
        raise ValueError("Changed protocol, stable algorithm or scoring strength")
    scenes = frozen_scenes(base)
    out.mkdir()
    reports = {}
    for name, make, options in (("stable", make_server, {}), ("spatial13", make_spatial_workbench, {"enhanced": True})):
        root = out / name
        root.mkdir()
        reports[name] = {}
        with serving(make(root / "session", **options)) as server:
            if name == "spatial13" and server.store.engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Changed checkpoint")
            for target, donor in TARGET_DONOR:
                result = export_cases(server, scenes, target, donor, root / target)
                reports[name][target] = result
                passed = sum(case["passed"] for case in result["cases"].values())
                print(f"{name} {target} from {donor}: {passed}/{len(result['cases'])}, "
                      f"identity={result['same_image']['exact']}, PNG exports={len(result['outputs']) + 1}", flush=True)
    json_to(out / "results.json", dict(indoor=PARENT, outdoor=OUTDOOR_SHA, indoor_bounds=ROOM_SHA,
            stable=STABLE_SHA, checkpoint=CHECKPOINT, strength=STRENGTH, script_sha=digest(code),
            protocol_sha=PROTOCOL_SHA, reports=reports))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
