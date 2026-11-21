"""Prepare photographed daylight controls and measure two-image reference transfer."""

from argparse import ArgumentParser
from hashlib import sha1, sha256
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from probe_capture_invariant_look import render_style, style_transforms
from probe_gros_film_b_cross import score
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)

SCENES = ("wall", "pillar")
TIMES = ("time0", "time5")
LUMA = np.array([.2126, .7152, .0722])
RELEASE = "91d49cdfa7bad201ecb7038a6030237a3331a7afd9ba52931281d89a05d0ef99"


def image(path):
    data = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None or data.ndim != 3 or data.shape[-1] != 3 or not np.isfinite(data).all():
        raise ValueError(f"Bad HDR photo: {path}")
    return data[..., ::-1].astype(np.float64)


def prepared_rgb(values, scale):
    positive = np.maximum(values * scale, 0)
    light = positive / (1 + positive)
    display = np.where(light <= .0031308, 12.92 * light, 1.055 * light**(1 / 2.4) - .055)
    return np.rint(display * 255).astype(np.uint8)


def pixel_info(array):
    return dict(dimensions=list(array.shape[:2][::-1]), negative=float((array < 0).mean()),
                rgb_min=float(array.min()), rgb_max=float(array.max()))


def prepare(source, out):
    release = checked_json(source / "release.json", RELEASE)
    transfers = {}
    for row in release["assets"]:
        path = source / row["path"]
        data = path.read_bytes()
        blob = sha1(f"blob {len(data)}\0".encode() + data).hexdigest() if path.suffix == ".json" else None
        if len(data) != row["size"] or (digest(path) != row["sha256"] if row["sha256"] else blob != row["blob"]):
            raise ValueError(f"Cannot verify {path}")
        transfers[row["path"]] = sha256(data).hexdigest()
    out.mkdir()
    cases = {}
    for name in SCENES:
        folder = source / "small-aligned" / name
        metadata = json.loads((folder / "meta.json").read_bytes())
        photos = {time: image(folder / "photo" / f"{time}_hdr.exr") for time in TIMES}
        if photos[TIMES[0]].shape != photos[TIMES[1]].shape:
            raise ValueError("Photo dimensions changed across time")
        scale = 4 / float(np.quantile(photos["time0"] @ LUMA, .95))
        environment = {}
        for time in TIMES:
            sample = image(folder / "envmap" / f"{time}_envmap.exr")
            sin_weight = np.sin(np.pi * (np.arange(len(sample)) + .5) / len(sample))
            env_rgb = np.average(sample.mean(1), weights=sin_weight, axis=0)
            environment[time] = pixel_info(sample) | {"sphere_mean_rgb": env_rgb.tolist()}
        frames = {time: prepared_rgb(photo, scale) for time, photo in photos.items()}
        paths = {time: save(out / f"{name}-{time}.png", rgb) for time, rgb in frames.items()}
        cases[name] = dict(scale=scale, photos={time: pixel_info(p) for time, p in photos.items()},
                           envmaps=environment, hashes=paths, alignment=metadata["alignment_method"],
                           hdr_method=metadata["hdr_method"],
                           mean_hdr_luma={time: float((photo @ LUMA).mean()) for time, photo in photos.items()},
                           comparison=panel(out / f"{name}-capture.png", {name: frames}))
    json_to(out / "inputs.json", dict(revision=release["revision"], license=release["license"],
                                      inputs=transfers, input_bytes=sum(row["size"] for row in release["assets"]), scenes=cases))


def run(source, out, prepared_sha, regions_sha):
    archive = checked_json(source / "inputs.json", prepared_sha)
    regions = checked_json(Path(__file__).with_name("wildrelight-wall-cross-regions.json"), regions_sha)
    photos = {scene: {time: checked_rgb(source / f"{scene}-{time}.png", row["hashes"][time]) for time in TIMES}
              for scene, row in archive["scenes"].items()}
    film = style_transforms(BASE)["film_b"]
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unfrozen renderer")
    out.mkdir()
    cases = {}
    for person, donor in (SCENES, SCENES[::-1]):
        target = photos[person]["time0"]
        expected = finish_spatial(target, render_style(target, film), STRENGTH)
        xy = regions[person]
        height, width = target.shape[:2]
        x0, y0, x1, y1 = xy["neutral"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Registered region outside image")
        roi = np.zeros((height, width), dtype=bool)
        roi[y0:y1, x0:x1] = True
        areas = dict(whole=np.ones_like(roi), neutral=roi)
        refs, frames, hashes = {}, {}, {}
        for light, time in zip(("cold", "warm"), TIMES, strict=True):
            base = photos[donor][time]
            refs[time] = {"off": base, "on": render_style(base, film)}
            frames[light] = {}
            for state, ref in refs[time].items():
                result = finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
                frames[light][state] = result
                hashes[f"{time}-{state}"] = save(out / f"{person}-from-{donor}-{time}-{state}.png", result)
        board = panel(out / f"{person}-comparison.png", {
            "target": {"original": target, "expected Film B": expected},
            **{f"reference {time}": ref for time, ref in refs.items()},
            **{f"13 result {time}": frames[light] for light, time in zip(("cold", "warm"), TIMES, strict=True)},
        })
        cases[person] = dict(reference_scene=donor, scores=score(target, expected, frames, areas),
                             board=board, hashes=hashes)
    protocol = Path(__file__).with_name("wildrelight-wall-cross-protocol.md")
    json_to(out / "results.json", dict(input_sha256=prepared_sha, regions_sha256=regions_sha, checkpoint=CHECKPOINT,
                                       strength=STRENGTH, protocol=digest(protocol), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("prepared_sha", nargs="?")
    parser.add_argument("regions_sha", nargs="?")
    options = parser.parse_args()
    if options.mode == "prepare":
        prepare(options.source, options.out)
    else:
        run(options.source, options.out, options.prepared_sha, options.regions_sha)
