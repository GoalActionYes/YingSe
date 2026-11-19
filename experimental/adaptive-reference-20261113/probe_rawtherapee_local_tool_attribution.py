"""Isolate two existing local tools in a pinned RawTherapee author profile."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from audit_rawtherapee_sky_pair import AREAS, BOXES, LUMA, read_png
from render_rawtherapee_sky_pair import SOURCES, check, disabled, render

REPORT_SHA = "3c8256292d0c92f11df08cc81c6bd865866f541188ee3de6339fc4b202045bf5"
CLI_SHA = "9172a03896f0200daf621909b024f6cd65a24837cad56d0748730b2fa6d2fb6a"
TOOLS = {"color_only": b"Expreti_0", "retinex_only": b"Expcolor_0"}


def audit_pair(before, after):
    h, w = before.shape[:2]
    scores = {}
    for area, (x0, y0, x1, y1) in zip(AREAS, BOXES["P1005231"], strict=True):
        box = np.s_[round(h*y0):round(h*y1), round(w*x0):round(w*x1)]
        delta = (after[box].astype(float) - before[box]) @ LUMA
        scores[area] = {"median_luma": float(np.median(delta)), "median_abs_luma": float(np.median(np.abs(delta)))}
    return scores


def run(source, cli, out, resume=False):
    source, cli, out = source.resolve(strict=True), cli.resolve(strict=True), out.resolve()
    check(cli, CLI_SHA)
    ext, raw_sha, pp3_sha = SOURCES["P1005231"]
    raw = source / f"P1005231.{ext}"
    check(raw, raw_sha)
    author = check(source / "P1005231-author.pp3", pp3_sha)
    parent = source / "single-thread"
    archived = json.loads(check(parent / "results.json", REPORT_SHA).decode())["cases"]["P1005231"]
    frames, profiles = {}, {}
    for name in ("off", "on"):
        filename = parent / name / f"P1005231-{name}.png"
        check(filename, archived["fingerprints"][name]["png"])
        frames[name], profiles[name] = read_png(filename)
    out.mkdir(exist_ok=resume)
    if (out / "results.json").exists():
        raise FileExistsError("Refusing to replace an existing report")
    results = {}
    for name, key in TOOLS.items():
        data = disabled(author, key)
        images, digests = [], []
        for trial in ("first", "repeat"):
            folder = out / trial
            folder.mkdir(exist_ok=True)
            config, png = folder / f"{name}.pp3", folder / f"{name}.png"
            if config.exists():
                if config.read_bytes() != data:
                    raise ValueError(f"Saved profile changed: {config.name}")
            else:
                if png.exists():
                    raise ValueError(f"Saved PNG has no associated profile: {png.name}")
                config.write_bytes(data)
            if not png.exists():
                env = dict(os.environ, RT_SETTINGS=str(folder/"settings"), RT_CACHE=str(folder/"cache"),
                           OMP_NUM_THREADS="1")
                render(cli, raw, config, png, env)
            rgb, profile = read_png(png)
            if rgb.shape != frames["off"].shape or profile != profiles["off"]:
                raise ValueError("Source dimensions or color profile changed")
            images.append(rgb)
            digests.append(sha256(png.read_bytes()).hexdigest())
        frames[name] = images[0]
        exact = bool(np.array_equal(*images))
        scores = audit_pair(frames["off"], images[0])
        strength = scores["sky"]["median_luma"] / archived["areas"]["sky"]["median_luma"]
        passed = (exact and scores["sky"]["median_luma"] <= -2 and strength >= .5
                  and all(scores[k]["median_abs_luma"] < 1 for k in ("foreground_1", "foreground_2")))
        results[name] = {"areas": scores, "exact_repeat": exact, "sky_fraction": strength, "passed": passed,
                         "png_sha256": digests, "pp3_sha256": sha256(data).hexdigest()}
    interaction = frames["on"].astype(np.int16) - frames["color_only"] - frames["retinex_only"] + frames["off"]
    report = {"parent": REPORT_SHA, "tools": results,
              "interaction": audit_pair(np.zeros_like(interaction), interaction),
              "passed": any(case["passed"] for case in results.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--resume", action="store_true")
    run(**vars(parser.parse_args()))
