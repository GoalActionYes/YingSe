"""Measure a documented local exposure edit through both frozen workbenches."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_arri_shared_look import metrics
from probe_gros_local_editor_channel import actual_png, frozen_json, request, serving
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, digest, json_to, panel
from reference_color_app.media import decode_image
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


SOURCE_RECEIPT = "e25e45f5dd6343d8186cbfc611debcc417e3a3121288ebafb14e1609b618cb6e"
CROP = (170, 70, 1400, 1140)
STATES = ("off", "on")


def prepare(source, output, evidence):
    paths = {state: source / state / "image.png" for state in STATES}
    if any(digest(path) != evidence["verification"]["png_file_hashes"][state]
           for state, path in paths.items()):
        raise ValueError("Author export differs from the frozen PNG")
    full = {state: decode_image(path) for state, path in paths.items()}
    if any(list(rgb.shape) != evidence["image_shape"] for rgb in full.values()):
        raise ValueError("Source raster differs from the frozen shape")

    x0, y0, x1, y1 = CROP
    crops = {state: output / f"crop-{state}.png" for state in STATES}
    for state, destination in crops.items():
        with Image.open(paths[state]) as image:
            image.crop(CROP).save(destination, icc_profile=image.info["icc_profile"])
        np.testing.assert_array_equal(decode_image(destination), full[state][y0:y1, x0:x1])

    rect_file = Path(__file__).with_name("playraw-flash-room-regions.json")
    saved = frozen_json(rect_file, evidence["verification"]["sample_regions_sha256"])
    boxes = {}
    for name, (left, top, right, bottom) in saved["rectangles"].items():
        if not (x0 <= left < right <= x1 and y0 <= top < bottom <= y1):
            raise ValueError(f"Reference crop excludes a registered region: {name}")
        boxes[name] = np.s_[top:bottom, left:right]
    for name in ("protected", "mug"):
        np.testing.assert_array_equal(full["off"][boxes[name]], full["on"][boxes[name]])
    return paths, crops, full, boxes


def added_endpoints(original, result):
    added = ((result == 0) & (original > 0)) | ((result == 255) & (original < 255))
    return float(np.any(added, axis=2).mean())


def evaluate(original, expected, off, on, boxes):
    result = {}
    for name, area in boxes.items():
        score = metrics(original, expected, off, on, area)
        passed = score["off_matches_original_mae"] <= 3 and score["on_matches_authored_mae"] <= 3
        passed = passed and (score["passed"] if name == "wanted" else score["response_mae"] <= 3)
        result[name] = dict(score, passed=passed)
    endpoints = {state: added_endpoints(original, rgb) for state, rgb in (("off", off), ("on", on))}
    return dict(regions=result, whole_image=metrics(original, expected, off, on, np.s_[:]),
                added_endpoints=endpoints, passed=all(r["passed"] for r in result.values())
                and max(endpoints.values()) <= .02)


def run(source, output):
    torch.set_num_threads(4)
    code = Path(__file__).resolve()
    evidence = frozen_json(code.parents[2] / "docs/research/2026-11-24-playraw-flash-room-evidence.json",
                           SOURCE_RECEIPT)
    if not evidence["qualified"] or STRENGTH != .8:
        raise ValueError("Expected qualified author edit and frozen strength")
    output.mkdir()
    paths, crops, actual, boxes = prepare(source, output, evidence)
    original = actual["off"]
    expected = finish_spatial(original, actual["on"], STRENGTH)
    if metrics(original, expected, original, expected, boxes["wanted"])["original_matches_authored_mae"] < 3:
        raise ValueError("Known edit did not survive the frozen 80% blend")
    rows = {"author": {"target off": original, "edit at 80%": expected, "cropped ref on": decode_image(crops["on"])}}
    engines = (("stable", make_server, {}), ("spatial13", make_spatial_workbench, {"enhanced": True}))
    scores = {}
    for name, builder, options in engines:
        with serving(builder(output / (name + "-data"), **options)) as server:
            if name == "spatial13" and server.store.engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Changed spatial checkpoint")
            owned = request(server, "/api/media", paths["off"].read_bytes(), "target.png")["id"]
            scores[name] = {}
            for framing, variants in (("full", paths), ("crop", crops)):
                exports, hashes = {}, {}
                for state, path in variants.items():
                    label = f"{name}-{framing}-{state}"
                    ref = request(server, "/api/media", path.read_bytes(), f"{framing}-{state}.png")["id"]
                    exports[state], hashes[state] = actual_png(server, owned, ref, output / (label + ".png"))
                    if exports[state].shape != original.shape:
                        raise ValueError("Workbench changed target dimensions")
                if framing == "full":
                    np.testing.assert_array_equal(exports["off"], original)
                scores[name][framing] = dict(evaluate(original, expected, exports["off"], exports["on"], boxes),
                                             output_sha256=hashes)
                rows[f"{name} {framing}"] = {"reference off": exports["off"], "reference on": exports["on"]}
                print(name, framing, scores[name][framing]["passed"], flush=True)

    json_to(output / "results.json", dict(stable=algorithm_fingerprint(), checkpoint=CHECKPOINT, strength=STRENGTH,
        source_evidence_sha256=SOURCE_RECEIPT, crop_box=CROP, crop_sha256={key: digest(path) for key, path in crops.items()},
        scores=scores, board=panel(output / "comparison.png", rows), script_sha256=digest(code),
        protocol_sha256=digest(code.with_name("playraw-flash-room-workbench-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
