"""Measure automatic cheek coverage against archived spectral correction arms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research"
PARENT = "1b7ebcf36dcb77b077cf5290e8c88a4558068492a8407eaaf74bcbabef933b41"
PEOPLE = ("235", "253", "282")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed input: {path}")
    return path


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def prepare(previews, output):
    from segment import SHA256, Segmenter

    previous = json.loads((DOCS / "2026-11-19-skin-spectral-portraits-evidence.json").read_bytes())
    donors = json.loads((DOCS / "2026-11-19-skin-spectral-donors-evidence.json").read_bytes())
    identities = previous["archive_sha256"] | donors["archive_sha256"]
    output.mkdir()
    detector = Segmenter()
    rows = {}
    try:
        for person in PEOPLE:
            name = f"{person}_RGB_pixelized.jpg"
            with Image.open(verified(previews / name, identities[name])) as image:
                rgb = np.asarray(image.convert("RGB"))
            probabilities, _ = detector.predict(rgb)
            mask = probabilities[..., 3] >= .70
            black = (rgb <= 3).all(axis=-1)
            if black.sum() < 1024:
                raise ValueError(f"Insufficient black support for {person}")
            path = output / f"{person}-face.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            rows[person] = {"preview_sha256": identities[name], "mask_sha256": digest(path),
                            "black_pixels": int(black.sum()), "black_selected": int(mask[black].sum())}
    finally:
        detector.close()
    write_json(output / "masks.json", {"detector_sha256": SHA256, "people": rows})


def sampled_gate(mask, bounds, patch, shape):
    output = np.zeros(shape, dtype=bool)
    for index, (left, top, right, bottom) in enumerate(bounds, 1):
        ys = np.rint(np.linspace(top, bottom - 1, patch)).astype(int)
        xs = np.rint(np.linspace(left, right - 1, patch)).astype(int)
        output[:patch, index * patch:(index + 1) * patch] = mask[np.ix_(ys, xs)]
    return output


def tile(image, row, column, patch):
    y = 27 + row * (2 * patch + 26)
    x = 8 + column * (4 * patch + 10)
    return image[y:y + 2 * patch, x:x + 4 * patch]


def evaluate(previews, masks, archive, spectra, output):
    from probe_skin_spectral_donors import scores
    from probe_skin_spectral_portraits import PATCH, compose, read_patches, render_panel, supports
    from probe_skin_spectrum_matrix import CQS, CQS_SHA, valid_in_both
    from probe_skin_spectrum_white import operators, table
    from segment import SHA256

    previous = json.loads(verified(archive / "results.json", PARENT).read_bytes())
    measurements = json.loads((masks / "masks.json").read_bytes())
    if measurements["detector_sha256"] != SHA256:
        raise ValueError("Unexpected detector")
    old = Path(__file__).with_name("skin-spectral-portraits-regions.json")
    extra = Path(__file__).with_name("skin-spectral-donors-regions.json")
    coords = json.loads(verified(old, previous["old_coordinates_sha256"]).read_bytes())
    coords.update(json.loads(verified(extra, previous["new_coordinates_sha256"]).read_bytes())["new_test"])
    white = json.loads((DOCS / "2026-11-19-skin-spectrum-white-evidence.json").read_bytes())
    for name, sha in white["sources"].items():
        verified(spectra / name, sha)
    tracked = json.loads((DOCS / "2026-11-19-skin-spectral-portraits-evidence.json").read_bytes())
    identities = tracked["archive_sha256"] | previous["archive_sha256"]
    with verified(spectra / CQS, CQS_SHA).open(encoding="utf-8-sig") as source:
        swatches = table(source).T
    transforms = operators(spectra)
    _, cheek, cores = supports()
    seam = cheek & ~np.logical_or.reduce(cores)
    result = {}
    output.mkdir()
    for person in PEOPLE:
        audit = measurements["people"][person]
        verified(previews / f"{person}_reflectance.mat", identities[f"{person}_reflectance.mat"])
        scene = compose(read_patches(previews, person, coords[person]), swatches)
        linear = {name: (scene @ matrix).reshape(-1, 3) for name, matrix in transforms.items()}
        valid = valid_in_both(linear).reshape(cheek.shape)
        with Image.open(verified(masks / f"{person}-face.png", audit["mask_sha256"])) as image:
            site = np.asarray(image) == 255
        gate = sampled_gate(site, coords[person], PATCH, cheek.shape)
        coverage = [float(gate[core & valid].mean()) for core in cores]
        black_rate = audit["black_selected"] / audit["black_pixels"]
        for direction in ("D65-A", "A-D65"):
            name = f"{person}-{direction}"
            frozen = previous["cases"][name]
            image_path = archive / f"{name}.png"
            with Image.open(verified(image_path, frozen["panel_sha256"])) as image:
                board = np.asarray(image.convert("RGB"))
            source, truth, baseline = (tile(board, 0, column, PATCH) for column in range(3))
            manual = tile(board, 1, 1, PATCH)
            prior = scores(manual, truth, valid, cores, seam)
            if prior != frozen["expanded"]:
                raise ValueError(f"Failed archived score replay for {name}")
            automatic = np.where(gate[..., None], manual, baseline)
            observed = scores(automatic, truth, valid, cores, seam)
            passed = (min(coverage) >= .95 and black_rate <= .01 and observed["gray_max"] <= 1
                      and observed["edge"]["p90"] <= prior["edge"]["p90"] + 1
                      and all(row["median"] <= 1.5 and row["p90"] <= 3 for row in observed["cheeks"]))
            path = output / f"{name}.png"
            render_panel(path, {"Spectral source": source, "Spectral target": truth, "Matrix": baseline,
                                "Registered cheeks": manual, "Preview gate": automatic})
            result[name] = {"coverage": coverage, "black_rate": black_rate, "manual": prior,
                            "automatic_preview_gate": observed, "passed": passed, "sha256": digest(path)}
    write_json(output / "results.json", {"parent_sha256": PARENT, "masks": measurements,
                                         "cases": result, "passed": all(x["passed"] for x in result.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, keys in {"prepare": ("previews", "output"),
                       "evaluate": ("previews", "masks", "archive", "spectra", "output")}.items():
        child = modes.add_parser(mode)
        for key in keys:
            child.add_argument(key, type=Path)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    (prepare if mode == "prepare" else evaluate)(**args)
