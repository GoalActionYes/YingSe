"""Prepare the fixed one-light target for the Beyond RGB source holdout."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_beyond_rgb_near_white import render
from probe_beyond_rgb_transfer import BATCH_CAPTURE_AUDIT
from probe_playraw_flash_room import PP3, RT_CLI
from probe_shared_local_warmth import prepare
from probe_target_mirror_chroma import checked_json, digest, json_to
from render_rawtherapee_sky_pair import check
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME, region_bounds


SCENE = "WHITE_green"
LIGHT = "1"
BLUE_TAGS_SHA = "9ac1bf804a0319ad4f2479376415a1d33b2b9004e7bdf5e9681fbbd7e856161e"


def d65_matrix(path):
    tags = json.loads(path.read_bytes())
    if int(tags["CalibrationIlluminant1"]) != 21:
        raise ValueError("Expected D65 camera calibration")
    return np.fromstring(tags["ColorMatrix1"].strip("[]"), sep=" ").reshape(3, 3)


def display(base, out):
    source = base / "beyond-rgb-batch-captures-v1"
    record = checked_json(source / "first.json", BATCH_CAPTURE_AUDIT)[SCENE]["captures"][LIGHT]
    folder = source / f"{SCENE}-light{LIGHT}"
    for name, expected in record["member_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Changed camera member: {name}")

    matrix = d65_matrix(folder / "oppo_tags.json")
    blue_tags = base / "beyond-rgb-near-white-v2/WHITE_blue-light1/oppo_tags.json"
    if digest(blue_tags) != BLUE_TAGS_SHA or not np.array_equal(matrix, d65_matrix(blue_tags)):
        raise ValueError("The two scenes must share the fixed D65 matrix")

    out.mkdir()
    receipt = render(folder, matrix, out / "light1.png")
    json_to(out / "display.json", dict(batch_camera_audit=BATCH_CAPTURE_AUDIT, scene=SCENE, light=LIGHT,
                                       members=record["member_sha256"], shared_d65_matrix=True, **receipt))


def grade(base, source, out, edit, guard):
    receipt = json.loads((source / "display.json").read_bytes())
    image_path = source / "light1.png"
    if receipt["scene"] != SCENE or receipt["light"] != LIGHT or digest(image_path) != receipt["sha256"]:
        raise ValueError("Changed companion target display")
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            raise ValueError("Expected RGB8 target")
        rgb = np.asarray(image)

    areas = region_bounds(rgb.shape, edit, [guard])
    out = out.absolute()
    cli = (base / RT_RUNTIME / "rawtherapee-cli.exe").absolute()
    if not cli.as_posix().isascii() or not out.as_posix().isascii():
        raise ValueError("Map the runtime and output directory to ASCII paths")
    check(cli, RT_CLI)
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    out.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    prepared = prepare(cli, template, icc, env, out / "render", {LIGHT: rgb}, areas, {LIGHT: True})
    json_to(out / "target.json", dict(source_display=receipt, editor=RT_CLI, template=PP3, profile=RT_PROFILE,
                                     **prepared))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    display_parser = actions.add_parser("display")
    display_parser.set_defaults(execute=display)
    grade_parser = actions.add_parser("grade")
    grade_parser.set_defaults(execute=grade)
    for command in (display_parser, grade_parser):
        command.add_argument("base", type=Path)
        command.add_argument("out", type=Path)
    grade_parser.add_argument("--source", type=Path, required=True)
    grade_parser.add_argument("--edit", nargs=4, type=int, required=True)
    grade_parser.add_argument("--guard", nargs=4, type=int, required=True)
    args = vars(parser.parse_args())
    args.pop("action")
    args.pop("execute")(**args)
