"""Replay a frozen skin correction on target-independent measured cheek textures."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
import torch

from probe_skin_spectrum_matrix import CQS, CQS_SHA, DIRECTIONS, matrix_render, valid_in_both
from probe_skin_spectrum_rgb_bound import MATRIX_SHA, render
from probe_skin_spectrum_white import WORKBOOK, cheek_samples, checksum, error_stats, mix, operators, table, to_linear, to_srgb
from spatial_render import SpatialEngine, finish_spatial


ARCHIVE_MD5 = {
    "253_reflectance.mat": "c061b49c03022740a235ccda4b9eaf54",
    "282_reflectance.mat": "74df80dcbe83d78183344a1a400a84c5",
    "253_RGB_pixelized.jpg": "6f199b70b446aa8fbbfc247cb05b4413",
    "282_RGB_pixelized.jpg": "7678a3a3e60a118e5a0ae66ff9765f4d",
}
PATCH = 64
FEATHER = 8


def read_patches(input_dir, name, regions):
    data = loadmat(input_dir / f"{name}_reflectance.mat")
    data, = [v for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 33]
    with Image.open(input_dir / f"{name}_RGB_pixelized.jpg") as image:
        if image.size != data.shape[:2][::-1]:
            raise ValueError("Spectral image and public preview dimensions differ")
    patches = []
    for left, top, right, bottom in regions:
        if not (0 <= left < right <= data.shape[1] and 0 <= top < bottom <= data.shape[0]):
            raise ValueError("Cheek patch outside measured image")
        y = np.rint(np.linspace(top, bottom - 1, PATCH)).astype(int)
        x = np.rint(np.linspace(left, right - 1, PATCH)).astype(int)
        cheek = data[y[:, None], x[None, :], :31].astype(float)
        if not np.all(np.isfinite(cheek) & (cheek >= 0) & (cheek <= 1)):
            raise ValueError("Cheek has reflectance outside the declared range")
        patches.append(cheek)
    return patches


def compose(cheeks, swatches):
    def tile(spectrum):
        return np.broadcast_to(spectrum, (PATCH, PATCH, 31))

    color = [tile(.25 * swatches[i] + .75 * .32) for i in (0, 8)]
    gray = [tile(value) for value in (.08, .18, .32, .48)]
    return np.concatenate((np.hstack((color[0], *cheeks, color[1])), np.hstack(gray)))


def supports():
    x, y = np.meshgrid(np.arange(PATCH), np.arange(PATCH))
    distance = np.minimum.reduce([x, y, PATCH - 1 - x, PATCH - 1 - y])
    patch_weights = np.minimum(1, distance / FEATHER)
    weights = np.zeros((PATCH * 2, PATCH * 4))
    weights[:PATCH, PATCH:PATCH * 3] = np.tile(patch_weights, (1, 2))
    core = []
    for col in (1, 2):
        mask = np.zeros_like(weights, dtype=bool)
        mask[:PATCH, col * PATCH:(col + 1) * PATCH] = distance >= FEATHER
        core.append(mask)
    cheek = np.zeros_like(weights, dtype=bool)
    cheek[:PATCH, PATCH:PATCH * 3] = True
    return weights, cheek, core


def render_panel(path, arms):
    width, height = (PATCH * 4 + 10), (PATCH * 2 + 26)
    sheet = Image.new("RGB", (width * 3 + 10, height * 2 + 10), "#1d2026")
    canvas = ImageDraw.Draw(sheet)
    for i, (name, data) in enumerate(arms.items()):
        x, y = 8 + width * (i % 3), 8 + height * (i // 3)
        canvas.text((x, y), name, fill="white")
        sheet.paste(Image.fromarray(data), (x, y + 19))
    sheet.save(path)


@torch.inference_mode()
def run(portraits, source, coordinates, matrix_evidence, white_evidence, output):
    if checksum(matrix_evidence) != MATRIX_SHA or checksum(source / CQS) != CQS_SHA:
        raise ValueError("Unexpected calibration evidence")
    previous = json.loads(matrix_evidence.read_bytes())
    if checksum(white_evidence) != previous["prior_evidence_sha256"]:
        raise ValueError("Unexpected white evidence")
    initial = json.loads(white_evidence.read_bytes())
    for name, identity in initial["sources"].items():
        if checksum(source / name) != identity:
            raise ValueError(f"Calibration source changed: {name}")
    hashes = {}
    for name, identity in ARCHIVE_MD5.items():
        if checksum(portraits / name, "md5") != identity:
            raise ValueError(f"Unexpected source image {name}")
        hashes[name] = checksum(portraits / name)
    regions = json.loads(coordinates.read_bytes())
    if set(regions) != {"253", "282"} or any(len(bounds) != 2 for bounds in regions.values()):
        raise ValueError("Expected two registered cheeks for both fixed portraits")
    with (source / CQS).open(encoding="utf-8-sig") as color_table:
        swatches = table(color_table).T
    transform = operators(source)
    issa = np.concatenate(tuple(cheek_samples(source / WORKBOOK).values()))
    training = {name: issa @ model for name, model in transform.items()}
    train_ok = valid_in_both(training)
    if int(train_ok.sum()) != 2095:
        raise ValueError("Frozen training support changed")
    prepared = {}
    for name, bounds in regions.items():
        spectrum = compose(read_patches(portraits, name, bounds), swatches)
        linear = {key: spectrum @ model for key, model in transform.items()}
        valid = valid_in_both({key: value.reshape(-1, 3) for key, value in linear.items()}).reshape(spectrum.shape[:2])
        prepared[name] = (valid, {key: to_srgb(value) for key, value in linear.items()})

    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != initial["checkpoint_sha256"]:
        raise ValueError("13 checkpoint changed")
    weights, cheeks, cores = supports()
    seam = cheeks & ~np.logical_or.reduce(cores)
    gray = np.s_[PATCH:]
    output.mkdir()
    results = {}
    for name, donor in (("253", "282"), ("282", "253")):
        valid, images = prepared[name]
        coverage = [float(valid[mask].mean()) for mask in (*cores, seam)]
        if not np.all(valid[~cheeks]):
            raise ValueError("Control patch out of common gamut")
        for before, after in DIRECTIONS:
            key = f"{before}-{after}"
            matrix = np.asarray(previous["controls"][key]["matrix"])
            src = images[before]
            target = mix(src, images[after])
            reference = prepared[donor][1][after]
            native = finish_spatial(src, engine.prepare(src, reference).apply(src), .8)
            ordinary, _ = matrix_render(src, matrix)
            linear = to_linear(to_srgb(training[before][train_ok]))
            residual = training[after][train_ok] - linear @ matrix
            unrestricted, _ = render(src, matrix, linear, residual)
            localized = np.rint(ordinary + weights[..., None] * (unrestricted.astype(np.int16) - ordinary)).astype(np.uint8)
            different = int(np.count_nonzero(np.any(localized != ordinary, axis=2) & ~cheeks))
            arms = {"YingSe 13": native, "CQS matrix": ordinary,
                    "Whole image residual": unrestricted, "Cheek-only residual": localized}
            scores = {}
            for arm, image in arms.items():
                gray_error = np.abs(image[gray].astype(np.int16) - target[gray]).mean(axis=-1)
                scores[arm] = {"cheeks": [error_stats(image[mask & valid], target[mask & valid]) for mask in cores],
                               "inner_edge": error_stats(image[seam & valid], target[seam & valid]),
                               "gray_max": float(gray_error.max())}
            old, new = (scores[arm] for arm in ("CQS matrix", "Cheek-only residual"))
            passed = (min(coverage) >= .95 and not different and new["gray_max"] <= 1
                      and new["inner_edge"]["p90"] <= old["inner_edge"]["p90"] + 1
                      and all(b["median"] <= min(1.5, a["median"] + .5) and b["p90"] <= 3
                              for a, b in zip(old["cheeks"], new["cheeks"], strict=True)))
            path = output / f"{name}-{key}.png"
            render_panel(path, {"Source": src, "Target light (80%)": target, **arms})
            results[f"{name}-{key}"] = {"coverage": coverage, "scores": scores, "non_skin_changed": different,
                                         "passed": bool(passed), "panel_sha256": checksum(path)}
    evidence = {"cases": results, "passed": all(case["passed"] for case in results.values()), "archive_sha256": hashes,
                "coordinates_sha256": checksum(coordinates), "matrix_sha256": MATRIX_SHA,
                "source_sha256": checksum(Path(__file__)),
                "protocol_sha256": checksum(Path(__file__).with_name("skin-spectral-portraits-protocol.md"))}
    (output / "results.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("portraits", "source", "coordinates", "matrix_evidence", "white_evidence", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
