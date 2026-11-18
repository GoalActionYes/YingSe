"""Test whether two frozen, separate portrait donors improve held-out correction."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_skin_spectral_portraits import PATCH, compose, read_patches, render_panel, supports
from probe_skin_spectral_support import PORTRAIT_SHA, neighbors
from probe_skin_spectrum_matrix import CQS, CQS_SHA, DIRECTIONS, matrix_render, valid_in_both
from probe_skin_spectrum_rgb_bound import MATRIX_SHA, render
from probe_skin_spectrum_white import WORKBOOK, cheek_samples, checksum, error_stats, mix, operators, table, to_linear, to_srgb


SUPPORT_SHA = "969ae9081bf86cdd7a4393452c1882a5bb9063dbf22d0b36a056008fe89ed239"
EXTRA_MD5 = {
    "222_reflectance.mat": "dba827df8ed5e38bd3463eee0754d51f",
    "226_reflectance.mat": "c20aed408b48e17c720b7b68dd79ae70",
    "235_reflectance.mat": "713c36a8ffae1df352eb56ea923aa198",
    "222_RGB_pixelized.jpg": "d2d6049c52a46267934d06af48232cc1",
    "226_RGB_pixelized.jpg": "e34e6089e836dd27e55ffeed171e5bda",
    "235_RGB_pixelized.jpg": "143650a5f13a22a8c84076d6bfaa4d4e",
}


def apply(source, matrix, fit_rgb, before, after, ordinary, weights):
    training = to_linear(to_srgb(fit_rgb[before]))
    residual = fit_rgb[after] - training @ matrix
    changed, _ = render(source, matrix, training, residual)
    image = np.rint(ordinary + weights[..., None] * (changed.astype(np.int16) - ordinary)).astype(np.uint8)
    return image, training


def scores(image, expected, valid, cores, seam):
    return {
        "cheeks": [error_stats(image[core & valid], expected[core & valid]) for core in cores],
        "edge": error_stats(image[seam & valid], expected[seam & valid]),
        "gray_max": float(np.abs(image[PATCH:].astype(np.int16) - expected[PATCH:]).mean(axis=-1).max()),
    }


def run(spectra, portraits, previous, output):
    folder = Path(__file__).parent
    docs = folder.parents[1] / "docs/research"
    evidence = {}
    for name, sha in (("skin-spectral-support", SUPPORT_SHA), ("skin-spectral-portraits", PORTRAIT_SHA),
                      ("skin-spectrum-matrix", MATRIX_SHA)):
        path = docs / f"2026-11-19-{name}-evidence.json"
        if checksum(path) != sha:
            raise ValueError(f"Frozen evidence changed: {name}")
        evidence[name] = json.loads(path.read_bytes())
    parent = evidence["skin-spectral-portraits"]
    matrix_evidence = evidence["skin-spectrum-matrix"]
    support = evidence["skin-spectral-support"]["results"]
    white_path = docs / "2026-11-19-skin-spectrum-white-evidence.json"
    if checksum(white_path) != matrix_evidence["prior_evidence_sha256"] or checksum(spectra / CQS) != CQS_SHA:
        raise ValueError("Frozen calibration changed")
    for name, sha in json.loads(white_path.read_bytes())["sources"].items():
        if checksum(spectra / name) != sha:
            raise ValueError(f"Frozen spectral data changed: {name}")
    for name, sha in parent["archive_sha256"].items():
        if checksum(portraits / name) != sha:
            raise ValueError(f"Frozen test data changed: {name}")
    sources = {}
    for name, md5 in EXTRA_MD5.items():
        if checksum(portraits / name, "md5") != md5:
            raise ValueError(f"Unexpected archive file: {name}")
        sources[name] = checksum(portraits / name)
    old_coords = folder / "skin-spectral-portraits-regions.json"
    if checksum(old_coords) != parent["coordinates_sha256"]:
        raise ValueError("Original test coordinates changed")
    new_coords = folder / "skin-spectral-donors-regions.json"
    registration = json.loads(new_coords.read_bytes())
    if set(registration["donors"]) != {"222", "226"} or set(registration["new_test"]) != {"235"}:
        raise ValueError("Unexpected participant roles")
    tests = json.loads(old_coords.read_bytes()) | registration["new_test"]
    transforms = operators(spectra)
    issa = np.concatenate(tuple(cheek_samples(spectra / WORKBOOK).values()))
    train = {key: issa @ operator for key, operator in transforms.items()}
    valid = valid_in_both(train)
    if valid.sum() != 2095:
        raise ValueError("Frozen ISSA count changed")
    train = {key: values[valid] for key, values in train.items()}
    donors = []
    for subject, bounds in sorted(registration["donors"].items()):
        points = np.vstack([patch[4::8, 4::8].reshape(64, 31) for patch in read_patches(portraits, subject, bounds)])
        donors.append({key: points @ operator for key, operator in transforms.items()})
    donor_counts = [int(valid_in_both(values).sum()) for values in donors]
    expanded = {key: np.concatenate([train[key], *(d[key][valid_in_both(d)] for d in donors)]) for key in train}
    with (spectra / CQS).open(encoding="utf-8-sig") as file:
        swatches = table(file).T
    weights, cheek, cores = supports()
    core = np.logical_or.reduce(cores)
    seam = cheek & ~core
    output.mkdir()
    results = {}
    for person, bounds in tests.items():
        scene = compose(read_patches(portraits, person, bounds), swatches)
        linear = {key: scene @ operator for key, operator in transforms.items()}
        valid = valid_in_both({key: v.reshape(-1, 3) for key, v in linear.items()}).reshape(cheek.shape)
        coverage = [float(valid[mask].mean()) for mask in (*cores, seam)]
        if not np.all(valid[~cheek]):
            raise ValueError("Non-skin controls lost common gamut")
        for before, after in DIRECTIONS:
            key = f"{before}-{after}"
            case = f"{person}-{key}"
            src = to_srgb(linear[before])
            target = mix(src, to_srgb(linear[after]))
            matrix = np.asarray(matrix_evidence["controls"][key]["matrix"])
            ordinary = matrix_render(src, matrix)[0]
            old, old_fit = apply(src, matrix, train, before, after, ordinary, weights)
            new, new_fit = apply(src, matrix, expanded, before, after, ordinary, weights)
            if case in parent["cases"]:
                panel = previous / f"{case}.png"
                if checksum(panel) != parent["cases"][case]["panel_sha256"]:
                    raise ValueError("Parent panel changed")
                with Image.open(panel) as image:
                    np.testing.assert_array_equal(old, np.asarray(image)[181:309, 540:796])
            prior = scores(old, target, valid, cores, seam)
            observed = scores(new, target, valid, cores, seam)
            same = bool(np.array_equal(new[~cheek], ordinary[~cheek]))
            passed = (min(coverage) >= .95 and same and observed["gray_max"] <= 1
                      and observed["edge"]["p90"] <= prior["edge"]["p90"] + 1
                      and all(y["median"] <= min(1.5, x["median"] + .5) and y["p90"] <= 3
                              for x, y in zip(prior["cheeks"], observed["cheeks"], strict=True)))
            paired_support = []
            overlay = src.copy()
            for area in cores:
                mask = area & valid
                query = to_linear(src[mask])
                _, radius_before = neighbors(query, old_fit)
                ids, radius_after = neighbors(query, new_fit)
                threshold = support[key]["threshold"]
                selected = ids >= len(old_fit)
                paired_support.append({"pixels": len(ids), "old_covered": int((radius_before <= threshold).sum()),
                                       "new_covered": int((radius_after <= threshold).sum()),
                                       "any_donor": int(selected.any(axis=1).sum()), "donor_fraction": float(selected.mean())})
                overlay[mask] = np.where(selected.any(axis=1)[:, None], [110, 200, 171], src[mask])
            path = output / f"{case}.png"
            render_panel(path, {"Source": src, "Spectral target (80%)": target, "CQS matrix": ordinary,
                                "ISSA only": old, "ISSA + 2 donors": new, "Donor use (green)": overlay})
            results[case] = {"coverage": coverage, "issa": prior, "expanded": observed, "support": paired_support,
                             "non_skin_unchanged": same, "passed": passed, "panel_sha256": checksum(path)}
    record = {"cases": results, "passed": all(v["passed"] for v in results.values()), "donor_samples": donor_counts,
              "archive_sha256": sources, "old_coordinates_sha256": checksum(old_coords),
              "new_coordinates_sha256": checksum(new_coords), "support_sha256": SUPPORT_SHA,
              "source_sha256": checksum(Path(__file__)), "protocol_sha256": checksum(folder / "skin-spectral-donors-protocol.md")}
    (output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("spectra", "portraits", "previous", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
