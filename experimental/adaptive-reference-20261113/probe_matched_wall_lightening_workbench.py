"""Run both baseline workbenches on matched true-light and local-wall references."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_arri_shared_look import metrics
from probe_gros_local_editor_channel import actual_png, request, serving
from probe_mid_matched_wall_cast import box
from probe_playraw_flash_room import read
from probe_shared_authored_tone_admission import ROOM_SHA, ROOMS
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


PARENT = "51dfe0f5e9e13517b6f43b3e0c11678ef5d8b970c47940e4d386cd8be8e79aef"
SOURCE = "shared-authored-tone-admission-v1"


def load(source, room, frames, tag):
    light, state = tag.split("-")
    path = source / room / f"{tag}.png"
    rgb, _, identity = read(path, False)
    expected = frames[light]["files"][state]
    if any(identity[k] != expected[k] for k in ("pixels_sha256", "icc_sha256")):
        raise ValueError("Changed prepared room edit")
    return path, rgb


def scores(original, expected, outputs, areas):
    result = {}
    for name, area in areas.items():
        off, captured, edited = (outputs[k] for k in ("base", "captured", "edited"))
        row = metrics(original, expected, off, edited, area)
        transfer = float(np.abs(captured[area].astype(np.int16) - off[area]).mean())
        distinction = float(np.abs(captured[area].astype(np.int16) - edited[area]).mean())
        allowed = max(row["off_matches_original_mae"], row["on_matches_authored_mae"], row["response_mae"], transfer) <= 3
        result[name] = dict(row, captured_spread_rgb8=transfer, captured_edited_rgb8=distinction,
                            joint_passed=allowed and (name != "wall_left" or row["passed"]))
    return result


def render(server, source, baseline, ledger, out):
    traces = {}
    for target, donor in (ROOMS, ROOMS[::-1]):
        target_record = baseline[target]
        rooms, cases = baseline[donor]["renders"], baseline[donor]["comparisons"]
        root = out / target
        root.mkdir()
        before_path, before = load(source, target, target_record["renders"], "0-off")
        _, authored = load(source, target, target_record["renders"], "0-on")
        expected = finish_spatial(before, authored, STRENGTH)
        if not target_record["renders"]["0"]["metrics"]["admitted"]:
            raise ValueError("Target lamp not eligible")
        input_id = request(server, "/api/media", before_path.read_bytes(), before_path.name)["id"]
        own, own_sha = actual_png(server, input_id, input_id, root / "identical.png")
        self_control = dict(exact=bool(np.array_equal(before, own)), sha256=own_sha)
        areas = {name: box(before, bounds) for name, bounds in ledger[target]["boxes"].items()}
        y, x = np.ogrid[:before.shape[0], :before.shape[1]]
        x0, y0, x1, y1 = target_record["extent"]
        areas["outside"] = ~((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
        outputs, hashes, evidence = {}, {}, {}
        tiles = {"target": {"original": before, "expected": expected}}
        for direction, row in cases.items():
            if not row["passed"]:
                continue
            a, b = direction.split("-to-")
            use = dict(base=f"{a}-off", captured=f"{b}-off", edited=f"{a}-on")
            current = {}
            for role, tag in use.items():
                if tag not in outputs:
                    path, _ = load(source, donor, rooms, tag)
                    rid = request(server, "/api/media", path.read_bytes(), f"{donor}-{tag}.png")["id"]
                    output = root / f"{tag}.png"
                    outputs[tag], hashes[tag] = actual_png(server, input_id, rid, output)
                current[role] = outputs[tag]
                if current[role].shape != before.shape:
                    raise ValueError("Output dimensions changed")
            summary = scores(before, expected, current, areas)
            evidence[direction] = dict(reference_states=use, scores=summary,
                                       passed=self_control["exact"] and all(row["joint_passed"] for row in summary.values()))
            tiles[direction] = current
        board = panel(root / "comparison.png", tiles)
        traces[target] = dict(donor=donor, same_image=self_control, outputs=hashes, cases=evidence, board=board)
    return traces


def run(base, out):
    torch.set_num_threads(4)
    source = base / SOURCE
    admitted = checked_json(source / "results.json", PARENT)
    ledger = checked_json(base / "mid-gray-anchor-v1.json", ROOM_SHA)["scenes"]
    if admitted["matched"]["rooms"] != 12 or STRENGTH != .8:
        raise ValueError("Changed cohort or scoring strength")
    out.mkdir()
    reports = {}
    for name, make, args in (("stable", make_server, {}), ("spatial13", make_spatial_workbench, {"enhanced": True})):
        root = out / name
        root.mkdir()
        with serving(make(root / "session", **args)) as server:
            if name == "spatial13" and server.store.engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Changed checkpoint")
            reports[name] = render(server, source, admitted["rooms"], ledger, root)
    code = Path(__file__)
    json_to(out / "results.json", dict(parent=PARENT, stable=algorithm_fingerprint(), checkpoint=CHECKPOINT,
            strength=STRENGTH, reports=reports, script_sha=digest(code),
            protocol_sha=digest(code.with_name("matched-wall-lightening-workbench-protocol.md"))))
    print({model: {target: sum(r["passed"] for r in row["cases"].values()) for target,row in checks.items()}
           for model,checks in reports.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
