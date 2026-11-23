"""Replay a documented local darkening operation on real outdoor light states."""

from argparse import ArgumentParser
from itertools import permutations
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from probe_arri_shared_look import metrics
from probe_gros_author_tone_switch import PROFILE, section
from probe_gros_local_editor_channel import actual_png, request, serving
from probe_gros_mixed_light_transfer import difference
from probe_playraw_flash_room import LUMA, RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to, panel
from probe_wildrelight_wall_cross import SCENES, TIMES
from render_rawtherapee_sky_pair import check, disabled, render
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench

SOURCE = "7ef5465396069bf7d4ba342d58cd713495d088c5fe81bdff09320f2fc0cf8503"
REGIONS = "690db0ca2940354c985f99fa61be0485146eea08c2bcd005fce13d2d4340c0fc"
AUTHOR = "eb44da59228162de892a3cf1d7d0db38dc2d3744cf096d3bfd5b4b2b8916b899"
PROTECTED = {"wall": (200, 435, 350, 469), "pillar": (420, 375, 540, 430)}
STATES = ("off", "on")


def location(author, rect, shape):
    left, top, right, bottom = rect
    height, width = shape
    x, y = (left + right) / 2, (top + bottom) / 2
    rx, ry = 1.5 * (right - left), 2 * (bottom - top)
    source = section(author, b"Locallab")
    if b"\nName_1=" in source or b"\nLightness_0=-5" not in source:
        raise ValueError("Unexpected author edit")
    radius, = re.findall(rb"(?m)^Loc_0=([0-9;]+)\r?$", source)
    axes = np.fromstring(radius.decode().rstrip(";"), sep=";", dtype=int).reshape(2, 2)
    limits = (2000 * rx / width, 2000 * ry / height)
    radii = [round(n * limit / max(axis)) for axis, limit in zip(axes, limits) for n in axis]
    fields = dict(CenterX_0=str(round(2000 * (x / width - .5))), CenterY_0=str(round(2000 * (y / height - .5))),
                  Loc_0=";".join(map(str, radii)) + ";")
    for field, value in fields.items():
        key = field.encode() + b"="
        source, count = re.subn(rb"(?m)^" + key + rb"[^\r\n]*", key + value.encode(), source)
        if count != 1:
            raise ValueError(f"Unrecognized local geometry: {field}")
    return section(author, b"Version") + PROFILE + source, fields, (x - 2 * rx, y - 2 * ry, x + 2 * rx, y + 2 * ry)


def masks(shape, edit, protect, bound):
    y, x = np.ogrid[:shape[0], :shape[1]]

    def rectangle(rect):
        x0, y0, x1, y1 = rect
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    areas = dict(edit=rectangle(edit), protect=rectangle(protect), outside=~rectangle(bound))
    if any(not a.any() for a in areas.values()):
        raise ValueError("Missing measurement area")
    return areas


def quality(before, after, original, areas):
    error = np.abs(before.astype(np.int16) - original)
    delta = after.astype(np.int16) - before
    darken = float(np.median(delta[areas["edit"]] @ LUMA))
    guards = {name: float(np.percentile(np.abs(delta[areas[name]]).mean(axis=-1), 95)) for name in ("protect", "outside")}
    endpoints = added_endpoints(before, after)
    return dict(off_mae=float(error.mean()), off_max=int(error.max()), edit_delta_y_prime=darken,
                protected_rgb8_p95=guards,
                protected_changed_pixels={name: int((delta[areas[name]] != 0).any(axis=-1).sum()) for name in guards},
                added_endpoint_fraction=endpoints,
                passed=bool(error.max() <= 2 and error.mean() <= .25 and darken <= -3
                            and max(guards.values()) <= 1.5 and endpoints <= .02))


def prepare(base, cli, out):
    inputs = base / "wildrelight-wall-prepared-v1"
    archived = checked_json(inputs / "inputs.json", SOURCE)
    bounds = checked_json(Path(__file__).with_name("wildrelight-wall-cross-regions.json"), REGIONS)
    author = check(base / "rawtherapee-sky-pair-v1/R6_06283-author.pp3", AUTHOR)
    check(cli, RT_CLI)
    icc = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    env = dict(os.environ, RT_SETTINGS=str(out / "rt-settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    receipts, outputs, selected, tiles = {}, {}, {}, {}
    for scene in SCENES:
        photos = {time: checked_rgb(inputs / f"{scene}-{time}.png", archived["scenes"][scene]["hashes"][time])
                  for time in TIMES}
        shape = photos[TIMES[0]].shape[:2]
        if any(img.shape != (*shape, 3) for img in photos.values()):
            raise ValueError("Real lighting switch changed the image shape")
        edit = bounds[scene]["neutral"]
        profile, geometry, bound = location(author, edit, shape)
        selected[scene] = masks(shape, edit, PROTECTED[scene], bound)
        folder = out / scene
        folder.mkdir()
        profiles = {state: folder / f"{state}.pp3" for state in STATES}
        profiles["on"].write_bytes(profile)
        profiles["off"].write_bytes(disabled(profile))
        receipts[scene] = dict(geometry=geometry, outside_exclusion=bound, protect=PROTECTED[scene],
                               pp3_sha256={k: digest(p) for k,p in profiles.items()}, cases={})
        outputs[scene] = {}
        for time, original in photos.items():
            stamp = folder / time
            stamp.mkdir()
            source = stamp / "embedded-input.png"
            Image.fromarray(original).save(source, icc_profile=icc)
            paths = {state: stamp / f"{state}.png" for state in STATES}
            images, hashes = {}, {}
            for state, path in paths.items():
                render(cli, source, profiles[state], path, env)
                rgb, _, info = read(path, False)
                if rgb.shape != original.shape:
                    raise ValueError("Editor changed geometry")
                images[state], hashes[state] = rgb, info
            if hashes["off"]["icc_sha256"] != hashes["on"]["icc_sha256"]:
                raise ValueError("Editor changed color space at the edit switch")
            row = quality(images["off"], images["on"], original, selected[scene])
            receipts[scene]["cases"][time] = dict(metrics=row, files=hashes)
            outputs[scene][time] = paths
            tiles[f"{scene} {time}"] = images
            print(scene, time, row, flush=True)
    return receipts, outputs, selected, panel(out / "editor-comparison.png", tiles)


def evaluate(paths, output, selected):
    torch.set_num_threads(4)
    results = {}
    with serving(make_spatial_workbench(output / "spatial13-data", enhanced=True)) as server:
        if server.store.engine.checkpoint_sha != CHECKPOINT or STRENGTH != .8:
            raise ValueError("Changed workbench or strength")
        for target, donor in permutations(SCENES):
            slug = f"{target}-from-{donor}"
            sources = paths[target][TIMES[0]]
            before = read(sources["off"], False)[0]
            expected = finish_spatial(before, read(sources["on"], False)[0], STRENGTH)
            tid = request(server, "/api/media", sources["off"].read_bytes(), sources["off"].name)["id"]
            frames, hashes = {}, {}
            for time in TIMES:
                frames[time], hashes[time] = {}, {}
                for state in STATES:
                    ref = paths[donor][time][state]
                    rid = request(server, "/api/media", ref.read_bytes(), ref.name)["id"]
                    frames[time][state], hashes[time][state] = actual_png(server, tid, rid, output / f"{slug}-{time}-{state}.png")
            summary = {}
            for key, area in selected[target].items():
                rows = {time: metrics(before, expected, *(frames[time][s] for s in STATES), area) for time in TIMES}
                lights = {state: difference(*(frames[t][state] for t in TIMES), area) for state in STATES}
                responses = [frames[t]["on"].astype(np.int16) - frames[t]["off"] for t in TIMES]
                spread = difference(*responses, area)
                errors = ("off_matches_original_mae", "on_matches_authored_mae", "response_mae")
                passed = (all(max(row[name] for name in errors) <= 3 and (key != "edit" or row["passed"])
                              for row in rows.values()) and max(r["rgb8_mae"] for r in (*lights.values(), spread)) <= 3)
                summary[key] = dict(scores=rows, light_spread=lights, response_spread=spread, passed=passed)
            board = panel(output / f"{slug}-comparison.png", {"target": {"off": before, "wanted": expected}, **frames})
            results[slug] = dict(regions=summary, hashes=hashes, comparison=board,
                                 passed=all(region["passed"] for region in summary.values()))
    return results


def run(base, cli, output):
    output.mkdir()
    receipts, paths, selected, board = prepare(base, cli, output)
    admitted = all(case["metrics"]["passed"] for scene in receipts.values() for case in scene["cases"].values())
    comparison = evaluate(paths, output, selected) if admitted else {}
    code = Path(__file__)
    json_to(output / "results.json", dict(admitted=admitted, inputs=SOURCE, coordinates=REGIONS, cli=RT_CLI,
            author=AUTHOR, receipts=receipts, editor_comparison=board, workbench=comparison,
            script_sha256=digest(code), protocol_sha256=digest(code.with_name("wildrelight-authored-darken-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for arg in ("base", "cli", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
