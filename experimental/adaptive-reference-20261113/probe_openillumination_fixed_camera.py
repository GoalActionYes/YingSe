"""Prepare fixed-camera captures and exercise ordinary cross-object grading."""

from argparse import ArgumentParser
from hashlib import sha1
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from probe_capture_invariant_look import render_style, style_transforms
from probe_gros_film_b_cross import score
from probe_gros_render_intent_projection import grade
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)
from triplet_color import to_srgb

PACK = "99f94ae7fc8cb2c7662eafbc403eda57d61db9e1b557188aabff454fd0c64812"
SCENES = ("obj_02_egg", "obj_21_cube", "obj_43_paper_ball")
LIGHTS = ("001", "013")
LUMA = np.array([.2126, .7152, .0722])


def verify_files(input_dir, release):
    hashes = {}
    for asset in release["assets"]:
        path = input_dir / asset["path"]
        payload = path.read_bytes()
        hash_value = digest(path)
        blob = sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        expected, actual = (asset["sha256"], hash_value) if asset["sha256"] else (asset["blob"], blob)
        if len(payload) != asset["size"] or actual != expected:
            raise ValueError(f"Release asset changed: {path}")
        hashes[asset["path"]] = hash_value
    return hashes


def prepare_photo(path, target_size):
    with Image.open(path) as original:
        fields = original.getexif().get_ifd(34665)
        tags = {name: str(fields.get(tag)) for name, tag in
                (("shutter", 33434), ("f_number", 33437), ("iso", 34855), ("white_balance_field", 41987))}
        converted, policy = to_srgb(original)
        size = original.size
        painted = np.array(converted.resize(target_size, Image.Resampling.BILINEAR))
        return painted, size, tags, policy


def prepare(input_dir, out):
    release = checked_json(input_dir / "release.json", PACK)
    hashes = verify_files(input_dir, release)
    out.mkdir()
    shapes = {}
    for subject in SCENES:
        source = input_dir / "lighting_patterns" / subject
        with Image.open(source / "output/obj_masks/CA2.png") as mask_map:
            width, height = mask_map.size
            size = tuple(round(768 * n / max(width, height)) for n in mask_map.size)
            mask = np.array(mask_map.resize(size, Image.Resampling.NEAREST)) > 0
        if mask.sum() < 1500 or (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
            raise ValueError(f"Object support or framing failed for {subject}")
        internal = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        if internal.sum() < 1500:
            raise ValueError("Insufficient object interior")
        colors, metadata, frames = {}, {}, {}
        for light in LIGHTS:
            picture = source / "Lights" / light / "raw_undistorted/CA2.JPG"
            rgb, dimensions, tags, policy = prepare_photo(picture, size)
            if dimensions != (width, height):
                raise ValueError("Image and mask dimensions differ")
            colors[light] = rgb
            metadata[light] = {"exif": tags, "input_policy": policy}
            frames[light] = save(out / f"{subject}-{light}.png", rgb)
        a, b = [rgb.astype(float) for rgb in colors.values()]
        valid_neighbours = internal[:, :-1] & internal[:, 1:]
        edges = [np.diff(rgb @ LUMA, axis=1)[valid_neighbours] for rgb in (a, b)]
        geometry = float(np.corrcoef(edges)[0, 1])
        if not np.isfinite(geometry):
            raise ValueError("Insufficient geometry signal")
        mask_path = out / f"{subject}-interior.png"
        mask_sha = save(mask_path, internal.astype(np.uint8) * 255)
        shapes[subject] = dict(native_size=[width, height], prepared_size=list(size), object_pixels=int(mask.sum()),
                               interior_count=int(internal.sum()), mask_sha256=mask_sha, frames=frames,
                               metadata=metadata, gradient_correlation=geometry,
                               input_differences={"whole": float(np.abs(a-b).mean()), "object": float(np.abs(a[internal]-b[internal]).mean())},
                               board=panel(out / f"{subject}-captures.png", {"camera CA2": colors}))
    json_to(out / "inputs.json", dict(revision=release["revision"], license=release["license"], input_hashes=hashes,
                                    total_bytes=sum(row["size"] for row in release["assets"]), objects=shapes))


def run(source, out, prepared_sha):
    archive = checked_json(source / "inputs.json", prepared_sha)
    photos, regions = {}, {}
    for subject in SCENES:
        record = archive["objects"][subject]
        photos[subject] = {light: checked_rgb(source / f"{subject}-{light}.png", sha)
                           for light, sha in record["frames"].items()}
        mask = source / f"{subject}-interior.png"
        if digest(mask) != record["mask_sha256"]:
            raise ValueError("Changed offline region")
        with Image.open(mask) as opened:
            regions[subject] = np.array(opened) > 0
    film = style_transforms(BASE)["film_b"]
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Wrong checkpoint")
    out.mkdir()
    results = {}
    for subject, donor in permutations(SCENES, 2):
        target = photos[subject]["001"]
        expected = finish_spatial(target, render_style(target, film), STRENGTH)
        field = regions[subject]
        areas = {"whole": np.ones_like(field), "object": field}
        guides, grades, hashes = {}, {}, {}
        key = f"{subject}-from-{donor}"
        for group, light in zip(("cold", "warm"), LIGHTS, strict=True):
            guide = photos[donor][light]
            guides[light] = {"off": guide, "on": render_style(guide, film)}
            grades[group] = {state: grade(engine, target, ref) for state, ref in guides[light].items()}
            for state, rgb in grades[group].items():
                hashes[f"{light}-{state}"] = save(out / f"{key}-{light}-{state}.png", rgb)
        results[key] = dict(scores=score(target, expected, grades, areas), hashes=hashes,
                            board=panel(out / f"{key}-comparison.png", {
                                "target": {"original": target, "expected": expected},
                                **{f"reference {light}": item for light, item in guides.items()},
                                **{f"13 result {light}": grades[group]
                                   for group, light in zip(("cold", "warm"), LIGHTS, strict=True)},
                            }))
    protocol = Path(__file__).with_name("openillumination-fixed-camera-protocol.md")
    json_to(out / "results.json", dict(prepared=prepared_sha, protocol=digest(protocol), checkpoint=CHECKPOINT,
                                       strength=STRENGTH, cases=results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("prepared_sha", nargs="?")
    options = parser.parse_args()
    if options.mode == "prepare":
        prepare(options.source, options.out)
    else:
        run(options.source, options.out, options.prepared_sha)
