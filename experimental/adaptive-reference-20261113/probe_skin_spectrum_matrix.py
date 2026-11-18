"""Calibrate on standard Munsell samples, then score held-out skin spectra."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_skin_spectrum_white import (
    LIGHTS, WORKBOOK, apply_gain, cheek_samples, checksum, error_stats, mix,
    operators, table, to_linear, to_srgb,
)


CQS = "CIE_srf_CQS_5nm.csv"
CQS_SHA = "ea607d6f7a93f4d18bb051eefc68ae2f4bbc948bb92526f5bf0624284047e18c"
BASELINE_SHA = "b47276cdf44d0892aff120563c6bebfe7dcc0c5063da1e3a3cb7f8679ec8f38b"
DIRECTIONS = tuple((src, dst) for src in LIGHTS for dst in LIGHTS if src != dst)


def fit(source, target, source_white, target_white):
    diagonal = np.diag(target_white / source_white)
    _, _, axes = np.linalg.svd(source_white[None, :], full_matrices=True)
    color_basis = axes[1:].T
    design = source @ color_basis
    weights, _, rank, singular = np.linalg.lstsq(design, target - source @ diagonal, rcond=None)
    if rank != 2:
        raise ValueError("Calibration patches do not support two chromatic axes")
    matrix = diagonal + color_basis @ weights
    np.testing.assert_allclose(source_white @ matrix, target_white, atol=1e-12, rtol=0)
    return matrix, singular.tolist()


def valid_in_both(rgb):
    values = np.stack(tuple(rgb.values()))
    return np.all((values >= .002) & (values <= .98), axis=(0, 2))


def matrix_render(source, matrix):
    converted = to_linear(source) @ matrix
    clipped = int(np.count_nonzero((converted < 0) | (converted > 1)))
    return mix(source, to_srgb(converted)), clipped


def cqs_holdout(calibration, white, source, target):
    valid = valid_in_both(calibration)
    src, dst = calibration[source], calibration[target]
    indices = np.arange(len(src))
    diagonal, predicted, expected = [], [], []
    for i in indices[valid]:
        train = indices != i
        matrix, _ = fit(src[train], dst[train], white[source], white[target])
        image = to_srgb(src[i])
        diagonal.append(apply_gain(image, white[target] / white[source]))
        predicted.append(matrix_render(image, matrix)[0])
        expected.append(mix(image, to_srgb(dst[i])))
    if not expected:
        return {"passed": False, "valid": 0}
    scores = {key: error_stats(np.stack(values), np.stack(expected))
              for key, values in (("white", diagonal), ("matrix", predicted))}
    old, new = scores["white"], scores["matrix"]
    return {"valid": int(valid.sum()), **scores,
            "passed": bool(valid.sum() >= 8 and new["median"] <= old["median"] and new["p90"] <= old["p90"] + 1)}


def run(inputs, previous, evidence, output):
    if checksum(evidence) != BASELINE_SHA or checksum(inputs / CQS) != CQS_SHA:
        raise ValueError("Frozen evidence or CQS file changed")
    prior = json.loads(evidence.read_bytes())
    for name, expected in prior["sources"].items():
        if checksum(inputs / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    operators_by_light = operators(inputs)
    white = {key: model.sum(axis=0) for key, model in operators_by_light.items()}
    with (inputs / CQS).open(encoding="utf-8-sig") as color_table:
        swatches = table(color_table).T
    if swatches.shape != (15, 31) or not np.all((swatches >= 0) & (swatches <= 1)):
        raise ValueError("Expected 15 CQS reflectance spectra")
    calibration = {key: swatches @ model for key, model in operators_by_light.items()}
    matrices, controls = {}, {}
    for source, target in DIRECTIONS:
        key = f"{source}-{target}"
        matrices[key], values = fit(calibration[source], calibration[target], white[source], white[target])
        controls[key] = {"svd": values, "matrix": matrices[key].tolist(),
                         "cqs_holdout": cqs_holdout(calibration, white, source, target)}

    groups = {}
    for group, spectra in cheek_samples(inputs / WORKBOOK).items():
        rgb = {key: spectra @ model for key, model in operators_by_light.items()}
        valid = valid_in_both(rgb)
        old = prior["origins"][str(group)]
        if int(valid.sum()) != old["in_gamut"] or len(valid) != old["subjects"]:
            raise ValueError("Skin analysis support changed")
        result = {"subjects": len(valid), "valid": int(valid.sum()), "qualified": old["qualified"], "scores": {}}
        for source, target in DIRECTIONS:
            key = f"{source}-{target}"
            src, dst = to_srgb(rgb[source][valid]), to_srgb(rgb[target][valid])
            expected = mix(src, dst)
            neutral, clipped = matrix_render(src, np.eye(3))
            np.testing.assert_array_equal(neutral, src)
            assert clipped == 0
            rendered, clipped = matrix_render(src, matrices[key])
            scores = error_stats(rendered, expected)
            result["scores"][key] = {"white": old["scores"][key]["gain"], "matrix": scores,
                                      "clipped_channels": clipped,
                                      "passed": scores["median"] <= 1.5 and scores["p90"] <= 3 and clipped == 0}
        result["passed"] = bool(valid.mean() >= .95 and all(v["passed"] for v in result["scores"].values()))
        groups[group] = result

    output.mkdir()
    charts = {}
    for name, old in prior["charts"].items():
        panel = previous / f"{name}.png"
        if checksum(panel) != old["files"][panel.name]:
            raise ValueError(f"Chart source changed: {name}")
        with Image.open(panel) as opened:
            original = opened.convert("RGB")
        source = np.asarray(original.crop((8, 25, 264, 281)))
        expected = np.asarray(original.crop((540, 25, 796, 281)))
        rendered, _ = matrix_render(source, matrices[name.split("-", 2)[2]])
        render_path = output / f"{name}-matrix.png"
        Image.fromarray(rendered).save(render_path)
        canvas = Image.new("RGB", (original.width + 266, original.height), "#1d2026")
        canvas.paste(original, (0, 0))
        ImageDraw.Draw(canvas).text((original.width + 2, 5), "White + Munsell matrix", fill="white")
        canvas.paste(Image.fromarray(rendered), (original.width + 2, 25))
        panel_path = output / f"{name}.png"
        canvas.save(panel_path)
        gray_max = float(np.abs(rendered[192:].astype(np.int16) - expected[192:]).mean(axis=2).max())
        charts[name] = {"skin": error_stats(rendered[:192], expected[:192]),
                        "gray": error_stats(rendered[192:], expected[192:]), "gray_max": gray_max,
                        "passed": gray_max <= 1, "files": {p.name: checksum(p) for p in (render_path, panel_path)}}
    result = {"cqs_sha256": CQS_SHA, "prior_evidence_sha256": BASELINE_SHA, "controls": controls,
              "origins": groups, "charts": charts,
              "gates": {"skin": all(row["passed"] for row in groups.values() if row["qualified"]),
                        "cqs": all(row["cqs_holdout"]["passed"] for row in controls.values()),
                        "gray": all(value["passed"] for value in charts.values())},
              "source_sha256": checksum(Path(__file__)),
              "protocol_sha256": checksum(Path(__file__).with_name("skin-spectrum-matrix-protocol.md"))}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs", "previous", "evidence", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
