"""Test whether measured gray-ball color applies to all materials in a scene."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from probe_physical_blue_global_fit import display, linear, save, verified


EVIDENCE_SHA = "550471e06572eef53928c65e43f4e419d720b66bfcdbf51f5cba99d61c9b2f5b"
SCENES = ("main_experiment120", "kingston_dining10")
ANCHOR = 14
LUMA = np.array([.2126, .7152, .0722])
Y8 = np.array([.299, .587, .114])


def rgb(path, *, small=False):
    with Image.open(BytesIO(path)) as image:
        image = image.convert("RGB")
        if small:
            image.thumbnail((512, 512), Image.Resampling.BILINEAR)
        return np.array(image)


def valid(pixels):
    return np.all((pixels >= 16) & (pixels <= 239), axis=-1)


def chroma(pixels):
    r, g, b = np.moveaxis(linear(pixels).clip(1e-7), -1, 0)
    return np.stack((np.log(r / g), np.log(b / g)), axis=-1)


def gray_probe(pixels):
    if pixels.shape != (256, 256, 3):
        raise ValueError("Unexpected gray probe dimensions")
    y, x = np.ogrid[:256, :256]
    mask = ((y - 127.5)**2 + (x - 127.5)**2 <= 32**2) & valid(pixels)
    if mask.sum() < 512:
        return None
    return {"chroma": np.median(chroma(pixels)[mask], axis=0), "pixels": int(mask.sum())}


def apply(pixels, delta):
    if not delta.any():
        return pixels.copy(), 0.0
    image = linear(pixels)
    gain = np.exp([delta[0], 0, delta[1]])
    scaled = image * gain
    y = image @ LUMA
    unit = np.divide(y, scaled @ LUMA, out=np.ones_like(y), where=y > 0)
    result = scaled * unit[..., None]
    return display(result), float((result.max(axis=-1) > 1).mean())


def rms(signal):
    return float(np.sqrt(np.square(signal).mean()))


def report_regions(before, expected, corrected, material):
    p, q, s = (chroma(pixels) for pixels in (before, expected, corrected))
    wanted, generated = q - p, s - p
    common = valid(before) & valid(expected)
    y = before @ Y8
    keep_neutral = (np.ptp(before, axis=2) <= 20) & (y >= 40) & (y <= 220)
    choices = {"whole": (common, 1024), "neutral": (common & keep_neutral, 256)}
    choices.update({str(index): (common & (material == index), 256) for index in np.unique(material)})
    rows = {}
    for name, (mask, minimum) in choices.items():
        if mask.sum() >= minimum:
            rows[name] = {
                "pixels": int(mask.sum()), "uncorrected_rms": rms(wanted[mask]),
                "probe_rms": rms((generated - wanted)[mask]),
                "observed_mean_chroma": wanted[mask].mean(axis=0).tolist(),
            }
    return rows


def run(archive_root, evidence_file, out):
    evidence = json.loads(verified(evidence_file, EVIDENCE_SHA).read_bytes())
    out.mkdir()
    record = {}
    for scene in SCENES:
        path = verified(archive_root / f"{scene}.zip", evidence["archives"][scene])
        folder = out / scene
        folder.mkdir()
        cases = {}
        with ZipFile(path) as archive:
            def read(member):
                return archive.read(f"{scene}/{member}")

            before = rgb(read(f"dir_{ANCHOR}_mip2.jpg"), small=True)
            with Image.open(BytesIO(read("materials_mip2.png"))) as labels:
                if labels.mode != "P":
                    raise ValueError("Expected the author's indexed mask")
                material = np.array(labels.resize((before.shape[1], before.shape[0]), Image.Resampling.NEAREST))
            initial = gray_probe(rgb(read(f"probes/dir_{ANCHOR}_gray256.jpg")))
            if initial is None:
                raise ValueError(f"Invalid anchor probe: {scene}")
            for direction in range(25):
                expected = rgb(read(f"dir_{direction}_mip2.jpg"), small=True)
                probe = gray_probe(rgb(read(f"probes/dir_{direction}_gray256.jpg")))
                delta = probe["chroma"] - initial["chroma"] if probe else np.zeros(2)
                rendered, clipping = apply(before, delta)
                if direction == ANCHOR:
                    np.testing.assert_array_equal(rendered, before)
                regions = report_regions(before, expected, rendered, material)
                whole = regions.get("whole")
                eligible = bool(direction != ANCHOR and whole and whole["uncorrected_rms"] >= .05)
                severe = [key for key, item in regions.items() if key.isdigit()
                          and item["probe_rms"] - item["uncorrected_rms"] > .05]
                cases[str(direction)] = {
                    "probe_pixels": probe["pixels"] if probe else 0, "delta": delta.tolist(),
                    "clipped_fraction": clipping,
                    "regions": regions, "eligible": eligible, "material_regressions": severe,
                    "improved": bool(eligible and probe and whole["probe_rms"] <= .8 * whole["uncorrected_rms"]),
                    "sha256": save(folder / f"dir-{direction}.png", rendered),
                }
        evaluated = [item for item in cases.values() if item["eligible"]]
        wins = sum(item["improved"] for item in evaluated)
        record[scene] = {"eligible": len(evaluated), "improved": wins,
                         "severe_directions": sum(bool(item["material_regressions"]) for item in evaluated),
                         "passed": len(evaluated) >= 8 and wins >= .75 * len(evaluated)
                         and all(not item["material_regressions"] for item in evaluated),
                         "directions": cases}
    result = {"parent_sha256": EVIDENCE_SHA, "anchor": ANCHOR, "cases": record,
              "passed": all(item["passed"] for item in record.values())}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive_root", "evidence_file", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
