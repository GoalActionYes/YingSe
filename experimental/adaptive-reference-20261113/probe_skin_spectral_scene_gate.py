"""Check selective spectral correction on ordinary and warmed RGB photographs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from probe_skin_spectral_preview_gate import DOCS, digest, verified, write_json

RGB_CASES = "83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9"
REGIONS = "2e0fe1e6b0b47e3579263bbd7cee626713955fee6c3cde71ef8323784de702b1"
STATES = ("original", "warm")


def ledger(path, sha=None):
    return json.loads((verified(path, sha) if sha else path).read_bytes())


def image(path, sha=None, mode="RGB"):
    with Image.open(verified(path, sha) if sha else path) as source:
        if source.mode != mode:
            raise ValueError(f"Expected {mode} source: {path}")
        return np.array(source)


def save(folder, name, rgb):
    path = folder / f"{name}.png"
    Image.fromarray(rgb).save(path)
    return digest(path)


def calibration():
    from probe_skin_spectrum_rgb_bound import MATRIX_SHA
    report = ledger(DOCS / "2026-11-19-skin-spectrum-matrix-evidence.json", MATRIX_SHA)
    return {name: np.asarray(entry["matrix"]) for name, entry in report["controls"].items()}


def inputs(base, output):
    from probe_dast_local_edit_cross import INPUT_SHA
    from probe_skin_spectrum_matrix import matrix_render

    matrix = calibration()["D65-A"]
    pilot = ledger(DOCS / "2026-11-18-bg-real-pixel-pilot-evidence.json", RGB_CASES)
    parent = base / "background-intervention-20261116-v2"
    published = ledger(parent / "results.json", pilot["parent_sha256"])
    records = {row["pair"]: row["input_file"] for row in published["cases"]}
    originals = {}
    for entry in pilot["cases"].values():
        key = entry["target_pair"]
        source = records[key]
        originals[key] = image(parent / source["path"], source["sha256"])
    dast = base / "dast-exposure-prepared-v1"
    entries = ledger(dast / "inputs.json", INPUT_SHA)["groups"]
    for row in entries:
        key = row["group"]
        originals[f"dast-{key}"] = image(dast / str(key) / "nominal.png", row["frames"]["nominal"])
    output.mkdir()
    result = {}
    for key, rgb in originals.items():
        warmer, clipped = matrix_render(rgb, matrix)
        result[key] = {"original": save(output, f"{key}-original", rgb),
                       "warm": save(output, f"{key}-warm", warmer), "warm_clipped_channels": clipped}
    write_json(output / "inputs.json", result)


def locate(source, output):
    from segment import SHA256, Segmenter

    output.mkdir()
    result = {}
    model = Segmenter()
    try:
        for key, hashes in ledger(source / "inputs.json").items():
            result[key] = {}
            for state in STATES:
                name = f"{key}-{state}"
                rgb = image(source / f"{name}.png", hashes[state])
                confidence, _ = model.predict(rgb)
                mask = confidence[..., 3] >= .70
                result[key][state] = save(output, name, mask.astype(np.uint8) * 255)
    finally:
        model.close()
    write_json(output / "masks.json", {"model": SHA256, "images": result})


def training(spectra, portraits):
    from probe_skin_spectral_donors import EXTRA_MD5
    from probe_skin_spectral_portraits import read_patches
    from probe_skin_spectrum_matrix import valid_in_both
    from probe_skin_spectrum_white import WORKBOOK, cheek_samples, checksum, operators, to_linear, to_srgb
    from probe_skin_spectral_preview_gate import PARENT

    root = Path(__file__).parent
    previous = ledger(spectra.parent / "skin-spectral-donors/results.json", PARENT)
    white = ledger(DOCS / "2026-11-19-skin-spectrum-white-evidence.json")
    for name, sha in white["sources"].items():
        verified(spectra / name, sha)
    for name, md5 in EXTRA_MD5.items():
        if checksum(portraits / name, "md5") != md5:
            raise ValueError(f"Changed donor file: {name}")
    coords = ledger(root / "skin-spectral-donors-regions.json", previous["new_coordinates_sha256"])["donors"]
    batches = [np.vstack(tuple(cheek_samples(spectra / WORKBOOK).values()))]
    for person, bounds in sorted(coords.items()):
        batches.append(np.vstack([patch[4::8, 4::8].reshape(64, 31) for patch in read_patches(portraits, person, bounds)]))
    op = operators(spectra)
    sources, expected = [], []
    for batch in batches:
        rgb = {key: batch @ matrix for key, matrix in op.items()}
        valid = valid_in_both(rgb)
        sources.append(rgb["A"][valid])
        expected.append(rgb["D65"][valid])
    fit = to_linear(to_srgb(np.vstack(sources)))
    if len(fit) != 2351:
        raise ValueError("Changed spectral training set")
    return fit, np.vstack(expected)


def render(base, source, masks, spectra, portraits, output):
    from probe_dast_local_edit_cross import INPUT_SHA
    from probe_skin_spectrum_matrix import matrix_render
    from probe_skin_spectrum_rgb_bound import render as render_residual
    from segment import SHA256

    matrix = calibration()["A-D65"]
    fit, expected = training(spectra, portraits)
    residual = expected - fit @ matrix
    fixed = verified(Path(__file__).with_name("bg-real-skin-regions-protocol.md"), REGIONS)
    marks = re.findall(r"^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$",
                       fixed.read_text(encoding="utf-8"), re.MULTILINE)
    protected = {"雪地", "浅墙", "白球服"}
    face = {"面部小块", "前额", "后方人物脸颊"}
    dast = base / "dast-exposure-prepared-v1"
    entries = ledger(dast / "inputs.json", INPUT_SHA)["groups"]
    archived = {f"dast-{r['group']}": image(dast / str(r["group"]) / "nominal-mask.png",
                                             r["target_mask_sha256"], "L") > 0 for r in entries}
    mask_book = ledger(masks / "masks.json")
    if mask_book["model"] != SHA256:
        raise ValueError("Unexpected detector identity")
    output.mkdir()
    results, tests = {}, []
    for key, hashes in ledger(source / "inputs.json").items():
        maps, by_state = [], {}
        for state in STATES:
            name = f"{key}-{state}"
            rgb = image(source / f"{name}.png", hashes[state])
            mask = image(masks / f"{name}.png", mask_book["images"][key][state], "L") == 255
            ordinary = matrix_render(rgb, matrix)[0]
            final = ordinary.copy()
            if mask.any():
                final[mask] = render_residual(rgb[mask], matrix, fit, residual)[0]
            delta = np.abs(final.astype(np.int16) - ordinary)
            areas = {}
            for left_id, right_id, label, *coords in marks:
                if key == f"{left_id}-{right_id}":
                    left, top, right, bottom = map(int, coords)
                    tile = np.s_[top:bottom, left:right]
                    selected = float(mask[tile].mean())
                    peak = int(delta[tile].max())
                    areas[label] = {"selected": selected, "peak": peak, "rgb_mae": float(delta[tile].mean())}
                    if label in protected:
                        tests.append(selected == 0 and peak == 0)
                    if label in face:
                        tests.append(selected >= .95)
            if key in archived:
                reference = archived[key]
                coverage = float(mask[reference].mean())
                areas["archived_core"] = {"selected": coverage, "rgb_mae": float(delta[reference].mean())}
                tests.append(coverage >= .95)
            maps.append(mask)
            by_state[state] = {"selected": int(mask.sum()), "changed": int(np.any(delta, axis=-1).sum()),
                               "areas": areas, "baseline_sha256": save(output, f"{name}-matrix", ordinary),
                               "candidate_sha256": save(output, f"{name}-auto", final)}
        union = int(np.count_nonzero(maps[0] | maps[1]))
        results[key] = {"intersection_over_union": float(np.count_nonzero(maps[0] & maps[1]) / union) if union else None,
                        "states": by_state}
    if len(tests) != 22:
        raise ValueError("Missing qualification records")
    write_json(output / "results.json", {"cases": results, "passing_checks": sum(tests),
                                         "total_checks": len(tests), "passed": all(tests)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes_args = {"inputs": ("base", "output"), "locate": ("source", "output"),
                  "render": ("base", "source", "masks", "spectra", "portraits", "output")}
    for mode, names in modes_args.items():
        child = modes.add_parser(mode)
        for name in names:
            child.add_argument(name, type=Path)
    options = vars(parser.parse_args())
    task = options.pop("mode")
    {"inputs": inputs, "locate": locate, "render": render}[task](**options)
