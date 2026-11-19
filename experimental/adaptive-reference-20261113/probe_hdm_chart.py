"""Compare known Film A transfer on a photographed chart and two skin regions."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from probe_arri_shared_look import LINEAR, LOOK_SHA, prepare
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json, verified_rgb
from probe_sparks_shared_chroma import LUMA, chroma
from spatial_render import SpatialEngine, finish_spatial


SCENE = "hdr_testimage"
REF_SCENE = "sparks_4592"
REF_SHA = "34fb0882f660aa8aac7cb7d72fd10e47572d49f78f0a1107f3d6e9f87f6f3b60"
CHECKPOINT = "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9"


def prepare_chart(archive, cube, out):
    if digest(cube) != LOOK_SHA:
        raise ValueError("Unexpected Film A LUT")
    manifest = archive / "manifest.json"
    data = json.loads(manifest.read_bytes())
    raw = data["cases"][SCENE]["files"]["original"]
    if raw["frame"] != 1260 or not data["fetched"]:
        raise ValueError("Unexpected HDR Testimage source frame")
    out.mkdir()
    source = archive / raw["file"]
    _, audit, display = prepare({SCENE: (source, raw["sha256"], LINEAR, 1080, 1920)}, cube, out)
    json_to(out / "prepared.json", {"archive_sha256": digest(manifest), "look_sha256": LOOK_SHA,
                                   "chart": audit[SCENE], "display": display})


def get_reference(folder, role, expected):
    path = folder / f"{REF_SCENE}-{role}.png"
    if digest(path) != expected:
        raise ValueError(f"Changed reference {role}")
    with Image.open(path) as image:
        if image.size != (512, 270) or image.mode != "RGB":
            raise ValueError("Unexpected reference shape")
        return np.asarray(image)


def chart_metrics(images, mask):
    light = {key: rgb @ LUMA for key, rgb in images.items()}
    colors = {key: chroma(rgb, light[key]) for key, rgb in images.items()}
    expected = (colors["expected"] - colors["target"])[mask]
    response = (colors["13-on"] - colors["13-off"])[mask]
    power = float(np.sum(expected**2))
    projection = float(np.sum(response * expected) / power) if power else None
    mae = float(np.abs(response - expected).mean())
    return {
        "pixels": int(mask.sum()),
        "off_mean_chroma_shift": (colors["13-off"] - colors["target"])[mask].mean(axis=0).tolist(),
        "off_chroma_mae": float(np.abs((colors["13-off"] - colors["target"])[mask]).mean()),
        "known_chroma_change_mae": float(np.abs(expected).mean()),
        "response_projection": projection,
        "response_chroma_mae": mae,
        "luma_median": {key: float(np.median(value[mask])) for key, value in light.items()},
        "passed": projection is not None and .75 <= projection <= 1.25 and mae <= 3,
    }


def render_chart(prepared, ref_folder, rois, out):
    prep = verified_json(prepared / "prepared.json", rois["prepared_sha256"])
    files = prep["chart"]["images"]
    target, authored = (verified_rgb(prepared / f"{SCENE}-{role}.png", files[role]) for role in ("off", "on"))
    expected = finish_spatial(target, authored, .8)
    valid = ((target.min(-1) > 0) & (target.max(-1) < 255)
             & (authored.min(-1) > 0) & (authored.max(-1) < 255))
    masks = {}
    for name, points in rois["regions"].items():
        raster = Image.new("L", (512, 288))
        ImageDraw.Draw(raster).polygon([tuple(point) for point in points], fill=255)
        masks[name] = (np.asarray(raster) > 0) & valid
    alignment = prep["chart"]["alignment"]
    qualified = (alignment["zero"] >= .65 and (alignment["dx"], alignment["dy"]) == (0, 0)
                 and all(mask.sum() >= 32 for mask in masks.values()))
    out.mkdir()
    record = {"qualified": qualified, "prepared_sha256": rois["prepared_sha256"], "regions": rois["regions"],
              "valid_pixels": {name: int(mask.sum()) for name, mask in masks.items()}}
    if qualified:
        guide = verified_json(ref_folder / "results.json", REF_SHA)["prepared"][REF_SCENE]["images"]
        torch.set_num_threads(4)
        engine = SpatialEngine("multiref-v1")
        if engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Unexpected 13 checkpoint")
        displays = {"target": target}
        for role in ("off", "on"):
            reference = get_reference(ref_folder, role, guide[role])
            displays[f"13-{role}"] = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
        displays["expected"] = expected
        results = {name: chart_metrics(displays, mask) for name, mask in masks.items()}
        gray = [name for name in masks if name.startswith("gray")]
        ordered = sorted(gray, key=lambda name: results[name]["luma_median"]["target"])
        record.update(
            weight_sha256=CHECKPOINT, reference_sha256=guide, metrics=results, source_luma_order=ordered,
            gray_inversions={
                key: sum(results[a]["luma_median"][key] >= results[b]["luma_median"][key]
                         for a, b in zip(ordered, ordered[1:]))
                for key in displays
            },
        )
        board = Image.new("RGB", (1040, 170), "#222730")
        labels = ImageDraw.Draw(board)
        outputs = {}
        for column, (label, pixels) in enumerate(displays.items()):
            path = out / f"{label}.png"
            image = Image.fromarray(pixels)
            image.save(path)
            if label == "target":
                draw = ImageDraw.Draw(image)
                for points in rois["regions"].values():
                    contour = [tuple(point) for point in points]
                    draw.line((*contour, contour[0]), fill="#60e8ee")
            outputs[label] = digest(path)
            x = column * 260 + 2
            labels.text((x, 4), label, fill="white")
            board.paste(image.resize((256, 144), Image.Resampling.LANCZOS), (x, 20))
        board.save(out / "comparison.png")
        record.update(outputs_sha256=outputs, comparison_sha256=digest(out / "comparison.png"))
    json_to(out / "results.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    setup = commands.add_parser("prepare")
    for arg in ("archive", "cube", "out"):
        setup.add_argument(arg, type=Path)
    render = commands.add_parser("render")
    for arg in ("prepared", "reference", "rois", "out"):
        render.add_argument(arg, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_chart(args.archive, args.cube, args.out)
    else:
        render_chart(args.prepared, args.reference, json.loads(args.rois.read_bytes()), args.out)
