"""Measure known-white correction on color renders of measured cheek spectra."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
import torch

from spatial_render import SpatialEngine, finish_spatial


WORKBOOK = "ISSA_17_Jan_2025_Yan_Lu.xlsx"
CMF = "CIE_xyz_1931_2deg.csv"
LIGHTS = {"D65": "CIE_std_illum_D65.csv", "A": "CIE_std_illum_A_1nm.csv"}
SOURCES = dict(zip((WORKBOOK, CMF, LIGHTS["D65"], LIGHTS["A"]), (
    "67e15398f96dd9af08b5d209d2522e19", "17cca777db64b17170f06f67ce9d3ab7",
    "03d4eb9b837c60671627c946fb534deb", "ed0e4effb55d82b950c0912b6278a9d1",
)))
BANDS = np.arange(400, 701, 10)
RGB_FROM_XYZ = np.array([[3.2404542, -1.5371385, -.4985314], [-.9692660, 1.8760108, .0415560],
                         [.0556434, -.2040259, 1.0572252]])


def checksum(path, algorithm="sha256"):
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def cheek_samples(path):
    grouped = defaultdict(list)
    with ZipFile(path) as book:
        string_root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
        strings = ["".join(t.text or "" for t in element.findall(".//{*}t")) for element in string_root]
        with book.open("xl/worksheets/sheet2.xml") as sheet:
            bands = []
            for _, row in ElementTree.iterparse(sheet, events=["end"]):
                if not row.tag.endswith("}row"):
                    continue
                cells = {}
                for cell in row.findall("{*}c"):
                    column = cell.get("r").rstrip("0123456789")
                    value = cell.findtext("{*}v")
                    if value is not None and (column in {"A", "B", "C", "G", "I", "L"}
                                              or column in bands or row.get("r") == "2"):
                        cells[column] = strings[int(value)] if cell.get("t") == "s" else value
                if row.get("r") == "2":
                    labels = {int(v): k for k, v in cells.items() if v.isdigit()}
                    bands = [labels[wavelength] for wavelength in BANDS]
                elif (cells.get("G") == "2" and cells.get("I") == "SCI" and cells.get("A", "").isdigit()
                      and cells.get("L") == "10" and {"B", "C", *bands}.issubset(cells)):
                    reflectance = np.array([float(cells[band]) for band in bands]) / 100
                    if np.all(np.isfinite(reflectance) & (reflectance >= 0) & (reflectance <= 1)):
                        grouped[int(cells["B"]), cells["C"]].append(reflectance)
                row.clear()
    origins = defaultdict(list)
    for (origin, _), values in grouped.items():
        origins[origin].append(np.median(values, axis=0))
    return {origin: np.stack(values) for origin, values in sorted(origins.items())}


def table(path):
    data = np.loadtxt(path, delimiter=",")
    indices = np.searchsorted(data[:, 0], BANDS)
    np.testing.assert_array_equal(data[indices, 0], BANDS)
    return data[indices, 1:]


def operators(folder):
    observer = table(folder / CMF)
    result = {}
    for key, name in LIGHTS.items():
        spectral_xyz = observer * table(folder / name)
        result[key] = .65 * spectral_xyz @ RGB_FROM_XYZ.T / spectral_xyz[:, 1].sum()
    return result


def to_srgb(linear):
    linear = linear.clip(0, 1)
    encoded = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    return np.rint(255 * encoded).astype(np.uint8)


def to_linear(rgb):
    encoded = rgb.astype(float) / 255
    return np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)


def mix(original, edited):
    return np.rint(.2 * original + .8 * edited).astype(np.uint8)


def apply_gain(original, gain):
    return mix(original, to_srgb(to_linear(original) * gain))


def error_stats(original, reference):
    pixel_errors = np.abs(original.astype(np.int16) - reference).mean(axis=-1)
    return dict(zip(("median", "p90"), np.quantile(pixel_errors, (.5, .9)).tolist()))


def chart(rgb):
    return rgb.reshape(4, 4, 3).repeat(64, axis=0).repeat(64, axis=1)


def panel(folder, target_name, target, reference, expected, baseline, white):
    images = {"Source": target, "Reference": reference, "Spectral target (80%)": expected,
              "YingSe 13": baseline, "Known white gains": white}
    sheet = Image.new("RGB", (1336, 289), "#1d2026")
    draw = ImageDraw.Draw(sheet)
    hashes = {}
    for index, (name, image) in enumerate(images.items()):
        x = 8 + 266 * index
        draw.text((x, 5), name, fill="white")
        rendered = Image.fromarray(image)
        sheet.paste(rendered, (x, 25))
        if name in ("YingSe 13", "Known white gains"):
            path = folder / f"{target_name}-{'13' if name == 'YingSe 13' else 'white'}.png"
            rendered.save(path)
            hashes[path.name] = checksum(path)
    path = folder / f"{target_name}.png"
    sheet.save(path)
    hashes[path.name] = checksum(path)
    return hashes


@torch.inference_mode()
def run(inputs, folder):
    torch.set_num_threads(4)
    for name, expected in SOURCES.items():
        if checksum(inputs / name, "md5") != expected:
            raise ValueError(f"Unexpected source file: {name}")
    transform = operators(inputs)
    white = {name: model.sum(axis=0) for name, model in transform.items()}
    np.testing.assert_allclose(np.stack(list(white.values())) @ np.array([.2126729, .7151522, .0721750]),
                               .65, atol=2e-6)
    directions = tuple((a, b) for a in LIGHTS for b in LIGHTS if a != b)
    outcomes, eligible = {}, {}
    neutral = np.repeat(np.array([[.08, .18, .32, .48]]).T, len(BANDS), axis=1)
    for origin, spectra in cheek_samples(inputs / WORKBOOK).items():
        linear = {name: spectra @ model for name, model in transform.items()}
        valid = np.logical_and.reduce([np.all((rgb >= .002) & (rgb <= .98), axis=1) for rgb in linear.values()])
        encoded = {name: to_srgb(rgb[valid]) for name, rgb in linear.items()}
        record = {"subjects": len(spectra), "in_gamut": int(valid.sum()), "coverage": float(valid.mean()),
                  "qualified": len(spectra) >= 50, "scores": {}}
        if not valid.any():
            outcomes[origin] = record
            continue
        for before, after in directions:
            image = encoded[before]
            np.testing.assert_array_equal(apply_gain(image, np.ones(3)), image)
            observed = apply_gain(image, white[after] / white[before])
            reference = mix(image, encoded[after])
            baseline, predicted = error_stats(image, reference), error_stats(observed, reference)
            record["scores"][f"{before}-{after}"] = {"identity": baseline, "gain": predicted,
                                                      "passed": predicted["median"] <= 1.5 and predicted["p90"] <= 3}
        record["passed"] = (record["coverage"] >= .95 and all(d["passed"] for d in record["scores"].values()))
        outcomes[origin] = record
        if record["qualified"] and record["in_gamut"] >= 12:
            y = spectra[valid] @ (table(inputs / CMF)[:, 1])
            order = np.argsort(y, kind="stable")
            indices = order[np.rint(np.linspace(0, len(order) - 1, 12)).astype(int)]
            eligible[origin] = np.vstack([spectra[valid][indices], neutral])
    donors = list(eligible)[:2]
    if len(donors) < 2:
        raise ValueError("Insufficient color chart source groups")
    folder.mkdir()
    engine = SpatialEngine("multiref-v1")
    boards = {group: {name: chart(to_srgb(eligible[group] @ model)) for name, model in transform.items()}
              for group in donors}
    comparisons = {}
    for group, donor in (donors, donors[::-1]):
        for before, after in directions:
            source, reference = boards[group][before], boards[donor][after]
            expected = mix(source, boards[group][after])
            gains = apply_gain(source, white[after] / white[before])
            render = finish_spatial(source, engine.prepare(source, reference).apply(source), .8)
            name = f"source-{group}-{before}-{after}"
            comparisons[name] = {"reference_origin": donor,
                                 "skin": {"13": error_stats(render[:192], expected[:192]),
                                          "gain": error_stats(gains[:192], expected[:192])},
                                 "gray": {"13": error_stats(render[192:], expected[192:]),
                                          "gain": error_stats(gains[192:], expected[192:])},
                                 "files": panel(folder, name, source, reference, expected, render, gains)}
    result = {"sources": {name: checksum(inputs / name) for name in SOURCES},
              "checkpoint_sha256": engine.checkpoint_sha, "white": {k: v.tolist() for k, v in white.items()},
              "origins": outcomes, "charts": comparisons,
              "passed": all(item.get("passed", False) for item in outcomes.values() if item["qualified"]),
              "source_sha256": checksum(Path(__file__)),
              "protocol_sha256": checksum(Path(__file__).with_name("skin-spectrum-white-protocol.md"))}
    (folder / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    CLI = argparse.ArgumentParser(description=__doc__)
    CLI.add_argument("inputs", type=Path)
    CLI.add_argument("outputs", type=Path)
    args = CLI.parse_args()
    run(args.inputs, args.outputs)
