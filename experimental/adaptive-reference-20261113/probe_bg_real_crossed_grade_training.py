"""Test the signal in crossed background and grade labels, under equal training budgets."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from background_intervention import target_regions
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import COOL, mask_for
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, read
from reference_color_app.media import decode_image


FIT_SHA = "58a4cd10c1d7e3d43aa5cfe5da311459aea5664022dcf012282e551645ee1703"
HELD_SHA = "83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9"
EDITS = {"none": (0, 0), "cool": (1, 0), "warm": (-1, 0), "contrast": (0, 1)}


def headroom(image):
    return np.rint(image.astype(float)*219/255 + 18).astype(np.uint8)


def grade(image, coeffs, strength=1.0):
    tint, contrast = coeffs
    edited = np.rint(np.clip(127.5 + (image-127.5)*(1+.16*contrast) + tint*COOL, 0, 255))
    return np.rint(image + strength*(edited-image)).astype(np.uint8)


def describe(image):
    rgb = image.reshape(-1, 3).astype(float)
    y = rgb @ [.2126, .7152, .0722]
    signal = np.column_stack([rgb, y, rgb[:,2]-rgb[:,0]])
    return np.percentile(signal, [10, 30, 50, 70, 90], axis=0).ravel()


def feature(target, reference):
    return np.concatenate((reference, reference-target))


def fit(samples, truth):
    x, y = np.asarray(samples), np.asarray(truth)
    center, scale, intercept = x.mean(axis=0), x.std(axis=0), y.mean(axis=0)
    scale = np.where(scale == 0, 1, scale)
    z = (x-center)/scale
    w = np.linalg.solve(z.T@z + np.eye(x.shape[1]), z.T@(y-intercept))
    return center, scale, intercept, w


def pair(folder, items):
    frames = {}
    for role in ("C1", "F1"):
        description = items[role]
        file = folder/description["name"]
        if digest(file) != description["sha256"]:
            raise ValueError("Frozen reference image changed")
        frames[role] = headroom(decode_image(file))
    return frames


def scores(actual, truth, person, neutral):
    values = {}
    cold_origin = {ref: actual[ref]["none"].astype(float) for ref in actual}
    for style in EDITS:
        swap = np.abs(actual["C1"][style].astype(float)-actual["F1"][style])
        known = truth[style].astype(float)-truth["none"]
        directions = {}
        for role, data in actual.items():
            delta = data[style].astype(float)-cold_origin[role]
            response = float(np.sum(delta*known)/np.sum(known**2)) if style != "none" else None
            directions[role] = {"projection": response, "response_mae": float(np.abs(delta-known).mean()),
                                "target_mae": float(np.abs(data[style].astype(float)-truth[style]).mean())}
        values[style] = {"person_swap": float(swap[person].mean()) if person.sum() >= 256 else None,
                         "neutral_swap": float(swap[neutral].mean()) if neutral.sum() >= 256 else None,
                         "response": directions}
    passed = True
    for style, row in values.items():
        passed &= all(value <= 1.5 for value in (row["person_swap"], row["neutral_swap"]) if value is not None)
        if style != "none":
            passed &= all(.75 <= x["projection"] <= 1.25 and x["response_mae"] <= 3 for x in row["response"].values())
    return {"styles": values, "qualified_neutral": bool(neutral.sum() >= 256), "passed": bool(passed and person.sum() >= 256)}


def panel(path, actual):
    sheet = Image.new("RGB", (1064, 5*280+4), "#222730")
    draw = ImageDraw.Draw(sheet)
    for i, (label, data) in enumerate(actual.items()):
        for j, (style, image) in enumerate(data.items()):
            x, y = 8+j*264, 6+i*280
            draw.text((x, y), f"{label} / {style}", fill="white")
            sheet.paste(Image.fromarray(image), (x, y+20))
    sheet.save(path)


def run(output):
    training = BASE/"bg-real-color-directions-inputs"
    index = training/"intake.json"
    held = ROOT/"docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    parent = PARENT/"results.json"
    if (digest(index), digest(held), digest(parent)) != (FIT_SHA, HELD_SHA, PARENT_SHA):
        raise ValueError("Frozen cohorts changed")
    fit_data = json.loads(index.read_bytes())
    test_data = json.loads(held.read_bytes())["cases"]
    test_inputs = [row["input"] for row in test_data.values()]
    if set(fit_data) & set(test_data) or {e["sha256"] for row in fit_data.values() for e in row.values()} & {
        e["sha256"] for row in test_inputs for e in row.values()
    }:
        raise ValueError("Fitting and evaluation reference images overlap")
    frames = {key: pair(training, fit_data[key]) for key in sorted(fit_data)}
    targets = [describe(value["C1"]) for value in frames.values()]
    x = {"control": [], "crossed": []}
    queue = []
    for i, items in enumerate(frames.values()):
        base = targets[(i+1)%len(targets)]
        for coeff in EDITS.values():
            c, f = (feature(base, describe(grade(items[role], coeff))) for role in ("C1", "F1"))
            x["control"].extend([c, c])
            x["crossed"].extend([c, f])
            queue.extend([coeff, coeff])
    assert len(queue) == 96
    trained = {name: fit(rows, queue) for name, rows in x.items()}
    cases = {row["pair"]: row for row in json.loads(parent.read_bytes())["cases"]}
    output.mkdir()
    results = {}
    worst = 0.0
    for item in test_data.values():
        ident = item["target_pair"]
        row = cases[ident]
        source = read(row["input_file"])
        target = headroom(source)
        descriptor = describe(target)
        human = mask_for(row, ident.split("-")[0])
        neutral = target_regions(source, human)["neutral_proxy"] & human
        known = {style: grade(target, coeff, .8) for style, coeff in EDITS.items()}
        references = pair(BASE/"bg-real-pixel-inputs", item["input"])
        reference_stats = {role: {style: describe(grade(img, coeff)) for style, coeff in EDITS.items()}
                           for role, img in references.items()}
        output_panel, record = {"Prescribed edit": known}, {}
        for arm, (center, scale, intercept, weights) in trained.items():
            estimates, outputs = {}, {}
            for role, versions in reference_stats.items():
                estimates[role], outputs[role] = {}, {}
                for style, value in versions.items():
                    predicted = ((feature(descriptor, value)-center)/scale) @ weights + intercept
                    coefficients = np.clip(predicted, [-1, 0], [1, 1])
                    estimates[role][style] = coefficients.tolist()
                    outputs[role][style] = grade(target, coefficients, .8)
                output_panel[f"{arm} / {role}"] = outputs[role]
            record[arm] = {**scores(outputs, known, human, neutral), "estimated": estimates}
        for style in EDITS:
            for name in references:
                old = record["control"]["styles"][style]["response"][name]["target_mae"]
                new = record["crossed"]["styles"][style]["response"][name]["target_mae"]
                worst = max(worst, new-old)
        path = output/f"{ident}.png"
        panel(path, output_panel)
        results[ident] = {**record, "panel_sha256": digest(path)}
    totals = {key: sum(int(v[key]["passed"]) for v in results.values()) for key in trained}
    report = {"rows": len(queue), "cases": results, "totals": totals, "worst_regression": worst,
              "passed": totals["crossed"] >= 3 and totals["crossed"] > totals["control"] and worst <= 2,
              "source_sha256": digest(Path(__file__)), "protocol_sha256": digest(Path(__file__).with_name(
                  "bg-real-crossed-grade-training-protocol.md"))}
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
