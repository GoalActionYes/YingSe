"""Validate and render the one pre-selected Beyond RGB near-white pair."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re

import h5py
import numpy as np
from PIL import Image

from audit_beyond_rgb_captures import compare, measure
from audit_beyond_rgb_lab_index import SPLIT_SIZE
from fetch_beyond_rgb_captures import ROLES, extract
from probe_beyond_rgb_transfer import XYZ_TO_SRGB


SCENE = "WHITE_blue"
LIGHTS = ("1", "7")
SELECTION_SHA = "3903a49f060cd0580c8dea7a07d86951abf2beab700976daae9c8b195c3a8853"


def validate_members(folder, members):
    links = [members[role] for role in ROLES]
    first = links[0]["offset"]
    last = links[-1]["offset"] + links[-1]["compressed_bytes"] + 1023
    headers = (folder / "headers.txt").read_text()
    if not re.search(r"(?im)^HTTP/\S+\s+206\b", headers):
        raise ValueError("Expected a partial HTTP response")
    if not re.search(rf"(?im)^content-range: bytes {first}-{last}/{SPLIT_SIZE}\s*$", headers):
        raise ValueError("Unexpected capture range")
    data = (folder / "archive.range").read_bytes()
    if len(data) != last - first + 1:
        raise ValueError("Unexpected capture range length")

    hashes = {}
    for link in links:
        name = Path(link["name"]).name
        payload = extract(data, link["offset"] - first, link)
        (folder / name).write_bytes(payload)
        hashes[name] = sha256(payload).hexdigest()
    (folder / "members.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return len(data)


def render(folder, matrix, destination):
    with h5py.File(folder / "oppo.h5") as handle:
        rgb = handle["oppo"][1::4, 1::4]
    actual = rgb @ (XYZ_TO_SRGB @ np.linalg.inv(matrix)).T
    outside = np.any((actual < 0) | (actual > 1), axis=-1)
    linear = np.clip(actual, 0, 1)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    output = np.rint(encoded * 255).astype(np.uint8)
    Image.fromarray(output).save(destination)
    np.save(destination.with_suffix(".gamut.npy"), outside)
    return dict(sha256=sha256(destination.read_bytes()).hexdigest(),
                gamut_limited_fraction=float(outside.mean()))


def run(selection, source, output):
    raw = selection.read_bytes()
    if sha256(raw).hexdigest() != SELECTION_SHA:
        raise ValueError("Selection differs from the completed near-white audit")
    selected = json.loads(raw)["selected"][SCENE]
    shots, centers, matrices, folders, sizes = {}, [], [], {}, {}
    for light in LIGHTS:
        folder = source / f"{SCENE}-light{light}"
        sizes[light] = validate_members(folder, selected[light])
        shots[light], points, _ = measure(folder)
        tags = json.loads((folder / "oppo_tags.json").read_bytes())
        if int(tags["CalibrationIlluminant1"]) != 21:
            raise ValueError("Expected fixed D65 calibration")
        matrices.append(np.fromstring(tags["ColorMatrix1"].strip("[]"), sep=" ").reshape(3, 3))
        centers.append(points)
        folders[light] = folder

    report = dict(captures=shots, fetched_range_bytes=sizes,
                  comparison=compare(*(shots[light] for light in LIGHTS), centers),
                  shared_d65_matrix=np.array_equal(*matrices))
    output.mkdir()
    checks = report["comparison"]
    if checks["geometry_pass"] and checks["exposure_pass"] and report["shared_d65_matrix"]:
        report["rendered"] = {light: render(folders[light], matrices[0], output / f"light{light}.png")
                              for light in LIGHTS}
    (output / "source.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("selection", "source", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
