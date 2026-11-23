"""Replay one documented local tone operator on frozen physical lighting states."""
from argparse import ArgumentParser
from itertools import combinations, permutations
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from probe_arri_shared_look import metrics
from probe_gros_local_editor_channel import actual_png, frozen_json, read_frame, request, serving
from probe_gros_local_editor_cross import REGIONS, SIZE, areas_and_weight
from probe_gros_mixed_light_exposure_toggle import TRANSFER
from probe_gros_mixed_light_transfer import LEVELS, LOCAL_EDIT, difference
from probe_playraw_flash_room import LUMA, PP3, RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel
from render_rawtherapee_sky_pair import check, disabled, render
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench

STATES = ("off", "on")
PROTECTED = ("screen_left", "screen_right", "outside")
PROFILE = (b"[ICM]\nInputProfile=(embedded)\nWorkingProfile=ProPhoto\n"
           b"OutputProfile=RTv4_sRGB\nOutputProfileIntent=Relative\nOutputBPC=true\n\n")


def section(data, name):
    pattern = rb"(?ms)^\[" + name + rb"\]\r?\n.*?(?=^\[|\Z)"
    result, = re.findall(pattern, data)
    return result.rstrip() + b"\n\n"


def relocate(author, regions):
    original = section(author, b"Locallab")
    geometry = {}
    left, top, right, bottom = regions["cheek_right"]
    _, _, radius_x, radius_y = regions["ellipse"]
    geometry["CenterX_0"] = str(round(2000 * ((left + right) / (2 * 384) - .5)))
    geometry["CenterY_0"] = str(round(2000 * ((top + bottom) / (2 * 256) - .5)))
    old_radii, = re.findall(rb"(?m)^Loc_0=([0-9;]+)\r?$", original)
    horizontal, vertical = np.fromstring(old_radii.decode().rstrip(";"), sep=";", dtype=int).reshape(2, 2)
    cap = (radius_x / 2 * 2000 / 384, radius_y / 2 * 2000 / 256)
    radii = [round(k * maximum / max(axis)) for axis, maximum in zip((horizontal, vertical), cap) for k in axis]
    geometry["Loc_0"] = ";".join(map(str, radii)) + ";"
    revised = original
    for key, value in geometry.items():
        revised, count = re.subn(rb"(?m)^" + key.encode() + rb"=[^\r\n]*", key.encode() + b"=" + value.encode(), revised)
        if count != 1:
            raise ValueError(key)
    return section(author, b"Version") + PROFILE + revised, geometry


def admit(before, after, target, areas):
    controls = np.abs(before.astype(np.int16) - target)
    delta = after.astype(np.float64) - before
    brightness = delta @ LUMA
    guards = {name: float(np.quantile(np.abs(delta[areas[name]]).mean(axis=1), .95)) for name in PROTECTED}
    medians = {name: float(np.median(brightness[areas[name]])) for name in ("cheek_left", "cheek_right")}
    clipping = added_endpoints(before, after)
    passed = (controls.max() <= 2 and controls.mean() <= .25 and medians["cheek_right"] >= 3
              and max(guards.values()) <= 1.5 and clipping <= .02)
    return dict(passed=bool(passed), off_rgb8_mae=float(controls.mean()), off_rgb8_max=int(controls.max()),
                luma_medians=medians, protected_rgb8_p95=guards, added_endpoint_fraction=clipping)


def source_inputs(base):
    archive = base / "gros-mixed-light-transfer-repeat-v1"
    workbench = checked_json(archive / "results.json", TRANSFER)
    cold = checked_json(base / "gros-local-editor-channel-v2" / "results.json", LOCAL_EDIT)
    for person, stored in workbench["prepared_hashes"].items():
        target = base / "gros-local-editor-channel-v2" / person / "cold" / "off.png"
        if cold["intake"][person]["cold"]["hashes"]["off"] != stored["target"]:
            raise ValueError("Changed target relationship")
        frames = {"target": read_frame(target, stored["target"])}
        frames.update((level, read_frame(archive / f"reference-{person}-{level}.png", stored[level])) for level in LEVELS)
        yield person, frames, stored


def prepare(base, cli, output):
    check(cli, RT_CLI)
    author = check(base / "playraw-flash-room-intake-v1" / "martin.pp3", PP3)
    icc_path = cli.parent / "iccprofiles/output/RTv4_sRGB.icc"
    icc = icc_path.read_bytes()
    directory = Path(__file__).parent
    coords = frozen_json(directory / "gros-local-editor-cross-regions.json", REGIONS)
    receipts, prepared, areas, images = {}, {}, {}, {}
    env = dict(os.environ, RT_SETTINGS=str(output / "rt-settings"), RT_CACHE=str(output / "rt-cache"), OMP_NUM_THREADS="1")
    for subject, frames, sources in source_inputs(base):
        group = output / subject
        group.mkdir()
        areas[subject], weight = areas_and_weight(coords[subject], coords["screen"])
        areas[subject]["outside"] = weight == 0
        profile, location = relocate(author, coords[subject])
        profiles = {state: group / f"{state}.pp3" for state in STATES}
        profiles["on"].write_bytes(profile)
        profiles["off"].write_bytes(disabled(profile))
        receipts[subject] = dict(location=location, input_sha256=sources, cases={},
                                 pp3_sha256={s: digest(p) for s, p in profiles.items()})
        prepared[subject], images[subject] = {}, {}
        for name, original in frames.items():
            if original.shape != (SIZE[1], SIZE[0], 3):
                raise ValueError("Incorrect frozen geometry")
            folder = group / name
            folder.mkdir()
            source = folder / "input.png"
            Image.fromarray(original).save(source, icc_profile=icc)
            paths = {s: folder / f"editor-{s}.png" for s in STATES}
            outputs, info = {}, {}
            for state in STATES:
                render(cli, source, profiles[state], paths[state], env)
                outputs[state], _, info[state] = read(paths[state], False)
            if any(p.shape != original.shape for p in outputs.values()) or info["off"]["icc_sha256"] != info["on"]["icc_sha256"]:
                raise ValueError("Editor changed dimensions or output profile")
            prepared[subject][name] = paths
            images[subject][name] = outputs
            admission = admit(outputs["off"], outputs["on"], original, areas[subject])
            receipts[subject]["cases"][name] = dict(admission=admission, output_pixels={s: info[s]["pixels_sha256"] for s in STATES},
                                                 icc_sha256=info["off"]["icc_sha256"])
            print(subject, name, admission, flush=True)
    return prepared, receipts, areas, images


def score(target, intended, frames, areas):
    rows = {}
    changes = {level: on.astype(np.float64) - off for level, (off, on) in frames.items()}
    for name, area in areas.items():
        scores = {level: metrics(target, intended, *states, area) for level, states in frames.items()}
        response_delta = {f"{a}-{b}": difference(changes[a], changes[b], area) for a,b in combinations(LEVELS,2)}
        if name == "cheek_right":
            accurate = all(row["passed"] and max(row["off_matches_original_mae"], row["on_matches_authored_mae"]) <= 3
                           for row in scores.values())
        else:
            accurate = all(max(row["off_matches_original_mae"], row["on_matches_authored_mae"], row["response_mae"]) <= 3
                           for row in scores.values())
        off_spread = {f"{a}-{b}": difference(frames[a][0], frames[b][0], area) for a,b in combinations(LEVELS,2)}
        rows[name] = dict(scores=scores, response_delta=response_delta, off_spread=off_spread,
                          passed=accurate and all(row["rgb8_mae"] <= 3 for row in (*response_delta.values(), *off_spread.values())))
    return rows


def compare(output, paths, images, regions):
    evidence = {}
    with serving(make_spatial_workbench(output / "spatial13-data", enhanced=True)) as server:
        if server.store.engine.checkpoint_sha != CHECKPOINT or STRENGTH != .8:
            raise ValueError("Changed workbench or strength")
        for target, donor in permutations(paths):
            slug = f"{target}-from-{donor}"
            path = paths[target]["target"]["off"]
            wanted = images[target]["target"]
            target_rgb = wanted["off"]
            expected = finish_spatial(target_rgb, wanted["on"], STRENGTH)
            tid = request(server, "/api/media", path.read_bytes(), path.name)["id"]
            results, hashes, tiles = {}, {}, {"target": {"off": target_rgb, "author at 80%": expected}}
            for level in LEVELS:
                rows, hashes[level] = [], {}
                for state, reference in paths[donor][level].items():
                    rid = request(server, "/api/media", reference.read_bytes(), reference.name)["id"]
                    frame, fingerprint = actual_png(server, tid, rid, output / f"{slug}-{level}-{state}.png")
                    if frame.shape != target_rgb.shape:
                        raise ValueError("Workbench changed geometry")
                    rows.append(frame)
                    hashes[level][state] = fingerprint
                results[level] = rows
                tiles[level] = dict(zip(STATES, rows))
            scores = score(target_rgb, expected, results, regions[target])
            required = ("cheek_right", *PROTECTED)
            evidence[slug] = dict(scores=scores, passed=all(scores[k]["passed"] for k in required),
                                  output_sha256=hashes, comparison_sha256=panel(output / f"{slug}-comparison.png", tiles))
    return evidence


def run(base, cli, output):
    torch.set_num_threads(4)
    output.mkdir()
    paths, receipts, regions, images = prepare(base, cli, output)
    passed = all(row["admission"]["passed"] for subject in receipts.values() for row in subject["cases"].values())
    result = compare(output, paths, images, regions) if passed else {}
    tiles = {f"{person} {level}": images[person][level] for person in paths for level in ("target", *LEVELS)}
    code = Path(__file__)
    json_to(output / "results.json", dict(admitted=passed, receipts=receipts, workbench=result, checkpoint=CHECKPOINT,
        strength=STRENGTH, profile_sha256=PP3, cli_sha256=RT_CLI, board=panel(output / "editor-comparison.png", tiles),
        script_sha256=digest(code), protocol_sha256=digest(code.with_name("gros-author-tone-switch-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("base", "cli", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
