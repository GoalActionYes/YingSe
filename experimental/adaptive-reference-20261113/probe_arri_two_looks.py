"""Compare two official creative LUTs across three fixed real source families."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from probe_arri_shared_look import LINEAR, LOOK_SHA, SOURCES, SPARKS_SHA, digest, metrics, prepare
from probe_hdm_chart import CHECKPOINT
from probe_hdm_hdr_display import json_to
from spatial_render import SpatialEngine, finish_spatial


ASSIGNMENTS = {"poker_fullshot": "hdr_testimage", "hdr_testimage": "sparks_4592", "sparks_4592": "poker_fullshot"}
STATES = ("off", "film_a", "film_b")


def build_sources(base):
    chart_dir = base / "hdm-chart-frames-v1"
    chart = json.loads((chart_dir / "manifest.json").read_bytes())["cases"]["hdr_testimage"]["files"]["original"]
    if chart["sha256"] != "4309b691687b428f1243bdbba539673aa5dd908829a8ede10d199bbdc49d6fd3":
        raise ValueError("Unexpected chart frame")
    return {
        "poker_fullshot": (base / "hdm-hdr-2014-paired-stills/poker_fullshot-original.exr",
                           SOURCES["poker_fullshot"], LINEAR, 1080, 1920),
        "hdr_testimage": (chart_dir / chart["file"], chart["sha256"], LINEAR, 1080, 1920),
        "sparks_4592": (base / "sparks-archival-grade-retry-input/nam-4592.exr", SPARKS_SHA, "ACES2065-1", 2160, 4096),
    }


def prepare_looks(base, out):
    archive = base / "arri-second-film-assets-v1"
    receipt = json.loads((archive / "receipt.json").read_bytes())
    film_a = base / "arri-shared-look-assets-v1/film-a.cube"
    film_b = archive / "second-film.cube"
    if digest(film_a) != LOOK_SHA or digest(film_b) != receipt["member_sha256"] or digest(film_b) == LOOK_SHA:
        raise ValueError("Unexpected frozen look input")
    sources = build_sources(base)
    photos = {}
    audit = {}
    for role, cube in (("film_a", film_a), ("film_b", film_b)):
        folder = out / role
        folder.mkdir()
        photos[role], audit[role], _ = prepare(sources, cube, folder)
    groups = {}
    for scene in sources:
        a, b = (photos[role][scene][0] for role in ("film_a", "film_b"))
        np.testing.assert_array_equal(a["off"], b["off"])
        images = {"off": a["off"], "film_a": a["on"], "film_b": b["on"]}
        neutral = np.logical_and.reduce([
            target_regions(rgb, np.zeros(rgb.shape[:2], dtype=bool))["neutral_proxy"]
            for rgb in images.values()
        ])
        separation = float(np.abs(a["on"].astype(np.int16) - b["on"]).mean())
        aligned = all(row[scene]["alignment"]["zero"] >= .65 and
                      (row[scene]["alignment"]["dy"], row[scene]["alignment"]["dx"]) == (0, 0)
                      for row in audit.values())
        groups[scene] = (images, neutral, {"style_separation_rgb8": separation,
                                          "neutral_pixels": int(neutral.sum()), "qualified": aligned and separation >= 1.})
    return groups, receipt, audit


def chart_masks(root, target):
    path = root / "experimental/adaptive-reference-20261113/hdm-chart-pilot-regions.json"
    rois = json.loads(path.read_bytes())["regions"]
    valid = np.logical_and.reduce([(rgb.min(-1) > 0) & (rgb.max(-1) < 255) for rgb in target.values()])
    masks = {}
    for name, points in rois.items():
        raster = Image.new("L", (target["off"].shape[1], target["off"].shape[0]))
        ImageDraw.Draw(raster).polygon([tuple(point) for point in points], fill=255)
        selected = (np.asarray(raster) > 0) & valid
        if selected.sum() >= 32:
            masks[name] = selected
    return masks


def run(base, out):
    out.mkdir()
    groups, receipt, audit = prepare_looks(base, out)
    engine = None
    cases = {}
    for name, donor in ASSIGNMENTS.items():
        target, neutral, gate = groups[name]
        reference, _, ref_gate = groups[donor]
        if not (gate["qualified"] and ref_gate["qualified"]):
            continue
        if engine is None:
            torch.set_num_threads(4)
            engine = SpatialEngine("multiref-v1")
            if engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Unexpected 13 checkpoint")
        displays = {}
        hashes = {}
        for role in STATES:
            rgb = finish_spatial(target["off"], engine.prepare(target["off"], reference[role]).apply(target["off"]), .8)
            path = out / f"{name}-13-{role}.png"
            Image.fromarray(rgb).save(path)
            displays[role], hashes[role] = rgb, digest(path)
        expected = {role: finish_spatial(target["off"], target[role], .8) for role in STATES}
        regions = {"whole": np.ones(neutral.shape, dtype=bool)}
        if neutral.sum() >= 256:
            regions["common_neutral"] = neutral
        if name == "hdr_testimage":
            regions.update(chart_masks(Path(__file__).resolve().parents[2], target))
        results = {
            region: {
                f"{a}_to_{b}": metrics(expected[a], expected[b], displays[a], displays[b], mask)
                for a, b in (("off", "film_a"), ("off", "film_b"), ("film_a", "film_b"))
            } for region, mask in regions.items()
        }
        cases[name] = {"donor": donor, "output_sha256": hashes, "results": results}
        board = Image.new("RGB", (816, 516), "#222730")
        draw = ImageDraw.Draw(board)
        for row, (row_name, images) in enumerate((("reference", reference), ("13 (80%)", displays), ("expected", expected))):
            for col, role in enumerate(STATES):
                x, y = col * 272 + 6, row * 172
                draw.text((x, y + 4), f"{row_name} / {role}", fill="white")
                thumb = Image.fromarray(images[role])
                thumb.thumbnail((256, 144))
                board.paste(thumb, (x, y + 22))
        board_path = out / f"{name}-comparison.png"
        board.save(board_path)
        cases[name]["comparison_sha256"] = digest(board_path)
    json_to(out / "results.json", {
        "assignments": ASSIGNMENTS, "look_a_sha256": LOOK_SHA, "look_b": receipt,
        "sources": {name: values[2] for name, values in groups.items()}, "preparation": audit,
        "weight_sha256": None if engine is None else CHECKPOINT, "cases": cases,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.base, args.output)
