"""Measure transfer of a background-only desaturation shared by two PSBench tasks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from triplet_color import to_srgb


REVISION = "929d0710eb7c86a988f8a69a74619052b60ac293"
BASE_URL = f"https://huggingface.co/datasets/zyn1216/PSBench/resolve/{REVISION}"
TASKS = ("145", "248")
CHECKPOINT = "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9"
REC709 = np.array([.2126, .7152, .0722])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries(trees):
    items = {}
    for task in TASKS:
        tree = trees / f"{task}-tree.json"
        listing = {row["path"]: row for row in json.loads(tree.read_bytes())}
        for role in ("start", "end"):
            file = f"{task}_{role}.jpg"
            row = listing[f"{task}/{file}"]
            items[file] = {"path": row["path"], "bytes": row["size"], "sha256": row["lfs"]["oid"]}
    if sum(item["bytes"] for item in items.values()) > 5 * 1024**2:
        raise ValueError("Exceeds the fixed download budget")
    return items


def fetch(trees, out):
    import requests

    items = entries(trees)
    out.mkdir(exist_ok=True)
    with requests.Session() as session:
        for name, row in items.items():
            dest = out / name
            if dest.exists() and dest.stat().st_size == row["bytes"] and sha(dest) == row["sha256"]:
                continue
            data = bytearray()
            with session.get(f"{BASE_URL}/{row['path']}", stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > row["bytes"]:
                        raise ValueError(f"Unexpected length: {name}")
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed published JPEG: {name}")
            if dest.exists():
                raise FileExistsError(f"Refusing to replace {dest}")
            dest.write_bytes(data)


def load(folder, filename, item):
    path = folder / filename
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError(f"Wrong JPEG: {filename}")
    with Image.open(path) as jpeg:
        image, details = to_srgb(ImageOps.exif_transpose(jpeg))
        size = image.size
        image.thumbnail((512, 512), Image.Resampling.BILINEAR)
        return np.array(image), {"size": list(size), "color": details["policy"]}


def blur(value):
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(value, 2, mode="edge"), (5, 5))
    return windows.mean(axis=(-2, -1))


def luminance_chroma(rgb):
    y = rgb @ REC709
    return y, rgb - y[..., None]


def regions(before, after):
    y0, c0 = luminance_chroma(before)
    y1, c1 = luminance_chroma(after)
    y0, y1 = blur(y0), blur(y1)
    common = (blur(np.linalg.norm(c0, axis=-1)) >= 12) & (np.abs(y0 - y1) <= 20)
    common &= (y0 >= 32) & (y0 <= 223) & (y1 >= 32) & (y1 <= 223)
    return {
        "desaturated": common & (blur(np.linalg.norm(c1, axis=-1)) <= 3),
        "retained": common & (blur(np.linalg.norm(c1 - c0, axis=-1)) <= 4),
    }


def score(before, after, off, on, areas):
    (_, a), (_, b), (_, c), (_, d) = map(luminance_chroma, (before, after, off, on))
    expected, response = b - a, d - c
    result = {}
    for role, area in areas.items():
        want, got = expected[area], response[area]
        projection = float(np.vdot(want, got) / np.vdot(want, want)) if np.any(want) else None
        result[role] = {
            "pixels": int(area.sum()), "projection": projection,
            "response_mae": float(np.abs(got - want).mean()),
            "response_magnitude": float(np.abs(got).mean()),
            "absolute_chroma_errors": {tag: float(np.abs(values[area] - b[area]).mean())
                                       for tag, values in (("identity", a), ("off", c), ("on", d))},
        }
    removed, preserved = (result[key] for key in ("desaturated", "retained"))
    passes = (.75 <= removed["projection"] <= 1.25 and removed["response_mae"] <= 3
              and preserved["response_magnitude"] <= 1.5)
    return {"regions": result, "whole_luma_response": float(np.abs((on.astype(float) - off) @ REC709).mean()),
            "passed": passes}


def panel(path, rows, areas):
    width, height = 265, 215
    result = Image.new("RGB", (width * 3, height * 2), "#222730")
    overlay = np.zeros_like(rows["Target"])
    overlay[areas["desaturated"]] = [238, 74, 85]
    overlay[areas["retained"]] = [101, 185, 255]
    draw = ImageDraw.Draw(result)
    for index, (label, data) in enumerate((*rows.items(), ("Score areas (red / blue)", overlay))):
        x, y = index % 3 * width, index // 3 * height
        tile = ImageOps.contain(Image.fromarray(data), (width - 12, height - 28))
        draw.text((x + 8, y + 4), label, fill="white")
        result.paste(tile, (x + (width - tile.width) // 2, y + 22))
    result.save(path)
    return sha(path)


def run(trees, media, out):
    from probe_inretouch_registration import gradient, shifts
    from spatial_render import SpatialEngine, finish_spatial
    import torch

    items = entries(trees)
    pairs, case = {}, {}
    out.mkdir()
    for task in TASKS:
        names = [f"{task}_{role}.jpg" for role in ("start", "end")]
        (before, a), (after, b) = [load(media, name, items[name]) for name in names]
        case[task] = {"images": dict(zip(names, (a, b))), "eligible": False}
        pairs[task] = (before, after)
        if a["size"] != b["size"]:
            continue
        registration = shifts(gradient(before), gradient(after))
        regions_count = {k: int(v.sum()) for k, v in regions(before, after).items()}
        gate = (registration[0]["zero"] >= .90 and (registration[0]["dy"], registration[0]["dx"]) == (0, 0)
                and all(r["peak_gain"] <= .01 for r in registration[1:])
                and min(regions_count.values()) >= 256)
        case[task].update(registration=registration, region_pixels=regions_count, eligible=gate)
    results = {}
    if all(row["eligible"] for row in case.values()):
        torch.set_num_threads(4)
        engine = SpatialEngine("multiref-v1")
        if engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed model")
        for task, donor in (TASKS, TASKS[::-1]):
            target = pairs[task][0]
            paths = {}
            for state, guide in zip(("off", "on"), pairs[donor], strict=True):
                pixels = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                paths[state] = out / f"{task}-{state}.png"
                Image.fromarray(pixels).save(paths[state])
            truth = load(media, f"{task}_end.jpg", items[f"{task}_end.jpg"])[0]
            off, on = (np.asarray(Image.open(path)) for path in paths.values())
            areas = regions(target, truth)
            results[task] = {
                "reference": donor, "hashes": {state: sha(path) for state, path in paths.items()},
                **score(target, truth, off, on, areas),
                "panel_sha256": panel(out / f"{task}-comparison.png",
                                      {"Target": target, "Edited reference": pairs[donor][1], "Author output": truth,
                                       "13 (original reference)": off, "13 (edited reference)": on}, areas),
            }
    report = {"revision": REVISION, "input_files": items, "model_sha256": CHECKPOINT, "strength": .8,
              "checks": case, "results": results,
              "passed": len(results) == len(TASKS) and all(row["passed"] for row in results.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    download = sub.add_parser("fetch")
    rinse = sub.add_parser("run")
    for command in (download, rinse):
        command.add_argument("trees", type=Path)
        if command is rinse:
            command.add_argument("media", type=Path)
        command.add_argument("out", type=Path)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    (fetch if mode == "fetch" else run)(**args)
