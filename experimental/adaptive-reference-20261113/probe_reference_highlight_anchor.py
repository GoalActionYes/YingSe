"""Check whether bright-edge color-line intersections carry reference edit information."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_facd_filter_bank import classifier, descriptor
from probe_facd_two_image_baseline import STYLES, read_checked
from probe_hdm_hdr_display import LUMA, json_to
from probe_hdm_hdr_equal_luminance import encode, linear_rgb
from probe_target_mirror_chroma import BG_SHA, FACD_SHA, FACD_SOURCE_SHA, checked_json, checked_rgb

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/adaptive-reference-20261113"


def ray_samples(rgb):
    preview = Image.fromarray(rgb)
    preview.thumbnail((256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(preview)
    light = linear_rgb(pixels)
    strength = light @ LUMA
    total = light.sum(axis=-1)
    color = np.divide(light[..., [0, 2]], total[..., None], out=np.zeros((*strength.shape, 2)),
                      where=total[..., None] > 0)
    valid = (pixels.min(axis=-1) >= 4) & (pixels.max(axis=-1) <= 249)
    ys, xs = np.indices(strength.shape)
    origins, directions, marks, dark_marks = [], [], [], []
    for first, second in (
        ((slice(None), slice(None, -2)), (slice(None), slice(2, None))),
        ((slice(None, -2), slice(None)), (slice(2, None), slice(None))),
    ):
        swap = strength[second] > strength[first]
        bright, dark = np.maximum(strength[first], strength[second]), np.minimum(strength[first], strength[second])
        ray = (color[second] - color[first]) * np.where(swap, 1, -1)[..., None]
        length = np.linalg.norm(ray, axis=-1)
        use = (valid[first] & valid[second] & (bright >= .30) & (bright <= .85) & (bright - dark >= .05)
               & (length >= .009) & (length <= .12))
        origins.append(np.where(swap[..., None], color[second], color[first])[use])
        directions.append(ray[use] / length[use, None])
        marks.append(np.column_stack((np.where(swap, ys[second], ys[first])[use],
                                      np.where(swap, xs[second], xs[first])[use])))
        dark_marks.append(np.column_stack((np.where(swap, ys[first], ys[second])[use],
                                           np.where(swap, xs[first], xs[second])[use])))
    return (preview, np.concatenate(origins), np.concatenate(directions), np.concatenate(marks),
            np.concatenate(dark_marks))


def locate(rgb):
    preview, origin, ray, mark, _ = ray_samples(rgb)
    report, support = fit_rays(preview.size, origin, ray, mark)
    return report, preview, mark[support]


def fit_rays(size, origin, ray, mark):
    width, height = size
    bins = (4 * mark[:, 0] // height) * 4 + 4 * mark[:, 1] // width
    counts = np.bincount(bins, minlength=16)
    support = counts[bins] >= 16
    r = {"candidate_pairs": len(bins), "support_pairs": int(support.sum()),
         "support_tiles": int((counts >= 16).sum()), "usable": False}
    if r["support_pairs"] < 128 or r["support_tiles"] < 4:
        return {**r, "reason": "coverage"}, support

    origin, ray = origin[support], ray[support]
    weights = 1. / counts[bins[support]]
    weights /= weights.sum()
    normals = np.column_stack((-ray[:, 1], ray[:, 0]))
    system = normals.T @ (normals * weights[:, None])
    minimum = float(np.linalg.eigvalsh(system)[0])
    r["smallest_eigenvalue"] = minimum
    if minimum < .10:
        return {**r, "reason": "directions"}, support

    position = np.linalg.solve(system, normals.T @ (weights * np.sum(normals * origin, axis=1)))
    displacement = position - origin
    forward = float(weights[np.sum(ray * displacement, axis=1) >= 0].sum())
    residuals = np.abs(np.sum(normals * displacement, axis=1))
    order = residuals.argsort(kind="stable")
    distance = float(residuals[order[np.searchsorted(np.cumsum(weights[order]), .5)]])
    r.update(proposed_rb=position.tolist(), forward_fraction=forward, median_distance=distance)
    r["usable"] = bool((position >= 0).all() and position.sum() <= 1 and forward >= .55 and distance <= .025)
    if not r["usable"]:
        r["reason"] = "convergence"
    return r, support


def synthetic_check():
    illuminant = np.array([.29, .32, .39])
    materials = np.array([[.41, .33, .26], [.21, .41, .38], [.23, .27, .50], [.40, .39, .21]])
    image = np.zeros((256, 256, 3), dtype=np.float64)
    for i in range(4):
        y, x = np.indices((64, 64))
        specular = (.05 + .46 * ((x % 12) > 5)) * .62
        rgb = .80 * materials[i] + specular[..., None] * illuminant
        for row in range(4):
            image[64*row:64*(row+1), 64*i:64*(i+1)] = rgb
    encoded = encode(image)
    estimate, _, _ = locate(encoded)
    error = np.linalg.norm(np.subtract(estimate["proposed_rb"], illuminant[[0, 2]]))
    if not estimate["usable"] or error > .02:
        raise ValueError(f"Synthetic reflection estimate failed: {estimate}, distance={error:.6f}")
    flat, _, _ = locate(np.full((256, 256, 3), 180, dtype=np.uint8))
    if flat["usable"]:
        raise ValueError("Flat input was accepted")
    return {"known_rb": illuminant[[0, 2]].tolist(), "anchor_error": float(error), "fit": estimate,
            "flat_control": flat}


def add_annotation(image, mark, title, anchor):
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    for y, x in mark[::4]:
        draw.point((int(x), int(y)), fill="#63dbef")
    panel = Image.new("RGB", (272, 304), "#222730")
    text = ImageDraw.Draw(panel)
    panel.paste(drawn, ((272 - image.width)//2, 22 + (256 - image.height)//2))
    text.text((8, 4), title, fill="white")
    decision = "usable" if anchor["usable"] else anchor["reason"]
    text.text((8, 281), f"{decision} / {anchor['support_pairs']} pairs / {anchor['support_tiles']} tiles",
              fill="white")
    return panel


def facd():
    source = BASE / "facd-border-registration-v1"
    known = checked_json(BASE / "facd-two-image-baseline-v1" / "results.json", FACD_SHA)
    registration = checked_json(ROOT / "docs/research/2026-11-18-facd-border-registration-evidence.json",
                                FACD_SOURCE_SHA)
    refs = known["references"]
    if len(refs) != 7:
        raise ValueError("Expected seven frozen FACD groups")
    candidates, baseline, features = {}, {}, {}
    for group in refs:
        candidates[group], baseline[group], features[group] = {}, {}, {}
        for style in STYLES:
            image = read_checked(source, group, style, registration)
            found, _, _ = locate(image)
            candidates[group][style] = found
            base = descriptor(image)
            extra = [255 * value for value in found["proposed_rb"]] + [255] if found["usable"] else [0, 0, 0]
            baseline[group][style] = base
            features[group][style] = np.concatenate((base, extra))
    predictions = {}
    totals = {}
    for key, data in (("baseline", baseline), ("anchor", features)):
        predictions[key] = {}
        correct = 0
        for target, queries in refs.items():
            train = [group for group in refs if group not in (target, *queries)]
            if len(train) != 4:
                raise ValueError("Unexpected photo holdout")
            mean, scale, params, bias = classifier([data[group][style] for group in train for style in STYLES],
                                                   list(range(len(STYLES))) * len(train))
            guesses = {}
            for group in queries:
                guesses[group] = {}
                for style in STYLES:
                    score = ((data[group][style] - mean) / scale) @ params + bias
                    guess = STYLES[int(np.argmax(score))]
                    guesses[group][style] = guess
                    correct += style == guess
            predictions[key][target] = {"training_photos": train, "queries": guesses}
        totals[key] = correct
    if totals["baseline"] != 28:
        raise ValueError(f"Could not reproduce the classifier: {totals['baseline']}")
    covered = sum(all(row["usable"] for row in group.values()) for group in candidates.values())
    return {"anchors": candidates, "predictions": predictions, "correct": totals, "fully_covered": covered,
            "passed": totals["anchor"] >= 34 and covered >= 5}


def background(out):
    source = BASE / "bg-real-editor-look"
    inputs = checked_json(source / "results.json", BG_SHA)
    if len(inputs["cases"]) != 4:
        raise ValueError("Expected four frozen background swaps")
    report = {}
    for case, material in inputs["cases"].items():
        rows = {}
        board = Image.new("RGB", (4*272, 304), "#222730")
        for i, name in enumerate(("C1-off", "C1-on", "F1-off", "F1-on")):
            file = f"{name}-reference.png"
            image = checked_rgb(source / case / file, material["artifacts"][file])
            found, preview, mark = locate(image)
            rows[name] = found
            board.paste(add_annotation(preview, mark, f"{case} / {name}", found), (i*272, 0))
        panel_path = out / f"{case}.png"
        board.save(panel_path)
        result = {"anchors": rows, "panel_sha256": sha256(panel_path.read_bytes()).hexdigest(), "passed": False}
        report[case] = result
        if not all(x["usable"] for x in rows.values()):
            continue

        c0, c1, f0, f1 = (np.asarray(rows[name]["proposed_rb"]) for name in rows)
        shifts = (c1-c0, f1-f0)
        edits = [float(np.linalg.norm(shift)) for shift in shifts]
        swaps = [float(np.linalg.norm(x)) for x in (f0-c0, f1-c1)]
        cosine = float(shifts[0] @ shifts[1]) / (edits[0] * edits[1]) if all(edits) else None
        result.update(edit_distances=edits, swap_distances=swaps, edit_cosine=cosine,
                      passed=bool(cosine is not None and cosine >= .5 and min(edits) >= max(swaps) + .005))
    groups = sum(x["passed"] for x in report.values())
    return {"cases": report, "passing_cases": groups, "passed": groups >= 3}


def run(out):
    check = synthetic_check()
    out.mkdir()
    sdr, bg = facd(), background(out)
    json_to(out / "results.json", {"synthetic": check, "facd": sdr, "background": bg,
                                   "facd_baseline_sha256": FACD_SHA, "facd_source_sha256": FACD_SOURCE_SHA,
                                   "background_baseline_sha256": BG_SHA,
                                   "passed": sdr["passed"] and bg["passed"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
