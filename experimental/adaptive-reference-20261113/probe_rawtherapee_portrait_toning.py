"""Render and measure the Cuckoo author profile's one color-toning switch."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from probe_tst_authored_neutral_intent import get_color
from render_rawtherapee_sky_pair import check, disabled, render


RAW_HASH = "7fbb1b6b9509e3ebb81aa9e76354c09ff1bf3002f465ef0364fcab6f8595304d"
PROFILE_HASH = "e6ebe23fd1e09bd8b53c7791f8f08b3ad4ad63e4467d597243f42319dd203cb0"
CLI_HASH = "9172a03896f0200daf621909b024f6cd65a24837cad56d0748730b2fa6d2fb6a"
PATHS = (b"DarkFrame=/szeva", b"FlatFieldFile=/szeva")


def render_state(source, cli, out, state):
    source, cli, out = source.resolve(strict=True), cli.resolve(strict=True), out.resolve()
    original = source / "DSC00178.ARW"
    check(original, RAW_HASH)
    check(cli, CLI_HASH)
    author = check(source / "profile-check.pp3", PROFILE_HASH)
    off = disabled(author, section=b"ColorToning")
    selected = off if state.startswith("off") else author
    if state.endswith("clear-paths"):
        for entry in PATHS:
            if selected.count(entry) != 1:
                raise ValueError("Unexpected calibration file configuration")
            selected = selected.replace(entry, entry.split(b"=")[0] + b"=", 1)
    out.mkdir()
    profile = out / "profile.pp3"
    profile.write_bytes(selected)
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "cache"), OMP_NUM_THREADS="1")
    render(cli, original, profile, out / "image.png", env)


def read(path):
    with Image.open(path) as image:
        if image.mode != "RGB" or not image.info.get("icc_profile"):
            raise ValueError("Expected an ICC-tagged RGB output")
        return np.array(image), sha256(image.info["icc_profile"]).hexdigest()


def audit(base, regions, out):
    if out.exists():
        raise FileExistsError(out)
    boxes = json.loads(regions.read_bytes())
    if set(boxes) != {"cheek", "light_material", "green_background"}:
        raise ValueError("Unexpected region registration")
    images = {}
    identity = {}
    for state in ("off", "on", "off-repeat", "on-repeat", "off-clear-paths", "on-clear-paths"):
        path = base / state / "image.png"
        image, icc = read(path)
        images[state] = image
        identity[state] = {"bytes_sha256": sha256(path.read_bytes()).hexdigest(), "icc_sha256": icc}
        base_state = state.split("-")[0]
        if state != base_state:
            np.testing.assert_array_equal(images[base_state], image)
        if identity["off"]["icc_sha256"] != icc:
            raise ValueError("Color profile changed between exports")
    before, after = images["off"], images["on"]
    if before.shape != after.shape:
        raise ValueError("Inconsistent geometry")
    rows = {}
    for name, (x0, y0, x1, y1) in boxes.items():
        if not (0 <= x0 < x1 <= before.shape[1] and 0 <= y0 < y1 <= before.shape[0]):
            raise ValueError("Region outside the registered raster")
        first, second = (image[y0:y1, x0:x1] for image in (before, after))
        delta = get_color(second) - get_color(first)
        rows[name] = {"pixels": int(delta.shape[0] * delta.shape[1]),
                      "chroma_delta_abs_median": float(np.median(np.abs(delta).mean(axis=-1))),
                      "chroma_signed_median": np.median(delta, axis=(0, 1)).tolist(),
                      "rgb_signed_median": np.median(second.astype(float)-first, axis=(0, 1)).tolist()}
    passed = rows["cheek"]["chroma_delta_abs_median"] >= 2 and all(
        rows[name]["chroma_delta_abs_median"] <= 1.5 for name in ("light_material", "green_background"))
    report = {"source": {"raw_sha256": RAW_HASH, "profile_sha256": PROFILE_HASH, "cli_sha256": CLI_HASH},
              "shape": list(before.shape), "outputs": identity, "boxes": boxes, "regions": rows, "passed": passed}
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("render")
    export.add_argument("source", type=Path)
    export.add_argument("cli", type=Path)
    export.add_argument("out", type=Path)
    export.add_argument("state", choices=("off", "on", "off-repeat", "on-repeat", "off-clear-paths", "on-clear-paths"))
    assess = sub.add_parser("audit")
    assess.add_argument("base", type=Path)
    assess.add_argument("regions", type=Path)
    assess.add_argument("out", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (render_state if command == "render" else audit)(**args)
