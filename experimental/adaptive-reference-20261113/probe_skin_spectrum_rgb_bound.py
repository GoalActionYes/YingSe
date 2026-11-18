"""Separate exact RGB8 conflicts from learnable errors under two known illuminants."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_skin_spectrum_matrix import DIRECTIONS, valid_in_both
from probe_skin_spectrum_white import WORKBOOK, cheek_samples, checksum, error_stats, mix, operators, to_linear, to_srgb


MATRIX_SHA = "aa3e997c630fdb473ef6bc3250ad8df6bb3ea97650e969724d2830b930ef573f"
N_CLOSEST = 8


def collisions(source, target):
    _, inverse, counts = np.unique(source, axis=0, return_inverse=True, return_counts=True)
    total = maximum = 0.0
    worst = None
    above = {"1.5": 0, "3.0": 0}
    for key in np.flatnonzero(counts > 1):
        ids = np.flatnonzero(inverse == key)
        labels = target[ids].astype(np.int16)
        center = np.median(labels, axis=0)
        total += np.abs(labels - center).sum()
        separation = np.abs(labels[None] - labels[:, None]).mean(axis=-1)
        index = np.unravel_index(separation.argmax(), separation.shape)
        bound = separation[index] / 2
        for cutoff in above:
            above[cutoff] += int(bound > float(cutoff))
        if bound > maximum:
            worst = ids[list(index)]
            maximum = float(bound)
    summary = {"samples": len(source), "shared_rgb_keys": int(np.count_nonzero(counts > 1)),
               "samples_in_shared_keys": int(counts[counts > 1].sum()),
               "mae_lower_bound_all_samples": float(total / (3 * len(source))),
               "groups_above_half_separation": above, "max_half_separation": maximum}
    return summary, worst


def nearest8(query, samples, residuals):
    neighbors = np.empty_like(query)
    for first in range(0, len(query), 128):
        last = min(first + 128, len(query))
        distance = ((query[first:last, None] - samples[None]) ** 2).sum(axis=2)
        order = np.argsort(distance, axis=1, kind="stable")[:, :N_CLOSEST]
        neighbors[first:last] = residuals[order].mean(axis=1)
    return neighbors


def render(source, matrix, samples, residuals):
    rgb, inverse = np.unique(source.reshape(-1, 3), axis=0, return_inverse=True)
    if len(samples) < N_CLOSEST:
        raise ValueError("Insufficient samples from other origins")
    linear = to_linear(rgb)
    projected = (linear @ matrix + nearest8(linear, samples, residuals))[inverse].reshape(source.shape)
    clipped = int(np.count_nonzero((projected < 0) | (projected > 1)))
    return mix(source, to_srgb(projected)), clipped


def conflict_panel(path, source, expected, indices, reference):
    sheet = Image.new("RGB", (696, 380), "#1d2026")
    draw = ImageDraw.Draw(sheet)
    for x, label in ((8, "Case"), (92, "Input sRGB8"), (262, "Shared reference"), (496, "Spectral target (80%)")):
        draw.text((x, 10), label, fill="white")
    for number, i in enumerate(indices):
        y = 34 + 171 * number
        draw.text((8, y + 52), str(number + 1), fill="white")
        draw.rectangle((92, y, 238, y + 147), fill=tuple(source[i].tolist()))
        sheet.paste(reference.resize((148, 148), Image.Resampling.NEAREST), (262, y))
        draw.rectangle((496, y, 643, y + 147), fill=tuple(expected[i].tolist()))
    sheet.save(path)


def append_chart(path, old, prediction):
    sheet = Image.new("RGB", (old.width + 266, old.height), "#1d2026")
    sheet.paste(old, (0, 0))
    ImageDraw.Draw(sheet).text((old.width + 2, 5), "Other-origin residual", fill="white")
    sheet.paste(Image.fromarray(prediction), (old.width + 2, 25))
    sheet.save(path)


def run(inputs, previous, evidence, white_evidence, output):
    if checksum(evidence) != MATRIX_SHA:
        raise ValueError("Unexpected matrix evidence")
    prior = json.loads(evidence.read_bytes())
    if checksum(white_evidence) != prior["prior_evidence_sha256"]:
        raise ValueError("Unexpected white baseline evidence")
    initial = json.loads(white_evidence.read_bytes())
    for name, identity in initial["sources"].items():
        if checksum(inputs / name) != identity:
            raise ValueError(f"Input identity changed: {name}")
    transform = operators(inputs)
    origins = cheek_samples(inputs / WORKBOOK)
    groups = np.repeat(list(origins), [len(samples) for samples in origins.values()])
    spectra = np.vstack(list(origins.values()))
    rgb = {key: spectra @ model for key, model in transform.items()}
    valid = valid_in_both(rgb)
    sources = {light: to_srgb(array[valid]) for light, array in rgb.items()}
    labels = groups[valid]
    output.mkdir()
    results = {}
    for a, b in DIRECTIONS:
        key = f"{a}-{b}"
        src, dst = sources[a], sources[b]
        expected = mix(src, dst)
        overlap, worst = collisions(src, expected)
        baseline = np.asarray(prior["controls"][key]["matrix"])
        samples = to_linear(src)
        residuals = rgb[b][valid] - samples @ baseline
        measurements = {}
        for origin, measurements_for_origin in origins.items():
            held = labels == origin
            older = prior["origins"][str(origin)]
            subject_count = len(measurements_for_origin)
            if held.sum() != older["valid"] or subject_count != older["subjects"]:
                raise ValueError("Frozen sample support changed")
            current, clipped = render(src[held], baseline, samples[~held], residuals[~held])
            scores = error_stats(current, expected[held])
            old = older["scores"][key]
            measurements[origin] = {
                "qualified": older["qualified"], "valid": int(held.sum()), "subjects": subject_count,
                "white": old["white"], "matrix": old["matrix"], "residual": scores, "clipped_channels": clipped,
                "passed": bool(held.sum() / subject_count >= .95 and scores["median"] <= 1.5
                               and scores["p90"] <= min(3, old["matrix"]["p90"] + .5)
                               and clipped <= old["clipped_channels"]),
            }
        chart = f"source-1-{key}"
        initial_path, panel = previous / f"{chart}.png", output / f"{chart}.png"
        if checksum(initial_path) != prior["charts"][chart]["files"][initial_path.name]:
            raise ValueError("Frozen reference chart changed")
        with Image.open(initial_path) as image:
            old_chart = image.convert("RGB")
        board = np.asarray(old_chart.crop((8, 25, 264, 281)))
        keep = labels != 1
        predicted, _ = render(board, baseline, samples[keep], residuals[keep])
        target = np.asarray(old_chart.crop((540, 25, 796, 281)))
        gray_max = float(np.abs(predicted[192:].astype(np.int16) - target[192:]).mean(axis=2).max())
        chart_scores = {"skin": error_stats(predicted[:192], target[:192]),
                        "gray_extrapolation": error_stats(predicted[192:], target[192:]),
                        "gray_max": gray_max, "gray_passed": gray_max <= 1}
        append_chart(panel, old_chart, predicted)
        images = {panel.name: checksum(panel)}
        if worst is not None:
            path = output / f"conflict-{key}.png"
            reference = old_chart.crop((274, 25, 530, 281))
            conflict_panel(path, src, expected, worst, reference)
            images[path.name] = checksum(path)
        results[key] = {"collision": overlap, "origins": measurements, "files": images, "chart": chart_scores,
                        "qualified_passed": sum(int(v["passed"]) for v in measurements.values() if v["qualified"])}

    result = {"results": results, "matrix_sha256": MATRIX_SHA, "white_sha256": checksum(white_evidence),
              "source_sha256": checksum(Path(__file__)),
              "protocol_sha256": checksum(Path(__file__).with_name("skin-spectrum-rgb-bound-protocol.md"))}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs", "previous", "evidence", "white_evidence", "output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
