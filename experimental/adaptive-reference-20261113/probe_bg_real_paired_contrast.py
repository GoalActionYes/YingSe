"""Isolate contrast estimation in the frozen BG-REAL subject arm."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_bg_real_crossed_grade_training import (
    BASE, EDITS, HELD_SHA, PARENT, PARENT_SHA, ROOT,
    digest, grade, headroom, mask_for, pair, panel, read, scores, target_regions,
)
from probe_dast_exposure_response import save_report

SUBJECT_SHA = "87031724f7d40f9bd24e2a573a662257559520552b5f9f16507fea2f89916c6a"


def contrast_from_pair(before, after):
    slopes, support = [], []
    for x, y in zip(before.reshape(-1, 3).T, after.reshape(-1, 3).T, strict=True):
        valid = (x >= 2) & (x <= 253) & (y >= 2) & (y <= 253)
        x, y = x[valid].astype(float), y[valid].astype(float)
        if x.size < 1024 or x.std() < 1.:
            raise ValueError("Insufficient usable contrast observations")
        centered = x - x.mean()
        slopes.append(float(centered @ (y - y.mean()) / (centered @ centered)))
        support.append(int(x.size))
    coefficient = float(np.clip((np.median(slopes) - 1.) / .16, 0., 1.))
    return coefficient, {"slopes_rgb": slopes, "support_rgb": support}


def write_outputs(dest, renders):
    dest.mkdir()
    hashes = {}
    for role, images in renders.items():
        for style, rgb in images.items():
            path = dest / f"{role}-{style}.png"
            Image.fromarray(rgb).save(path)
            hashes[path.name] = digest(path)
    return hashes


def run(output):
    subject_dir = BASE / "bg-real-crossed-grade-subject"
    sources = (subject_dir / "results.json", ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json",
               PARENT / "results.json")
    if tuple(map(digest, sources)) != (SUBJECT_SHA, HELD_SHA, PARENT_SHA):
        raise ValueError("Frozen evidence changed")
    frozen, held, parent = (json.loads(path.read_bytes())["cases"] for path in sources)
    targets = {row["pair"]: row for row in parent}
    output.mkdir()
    records = {}
    worst = 0.
    for case in held.values():
        key = case["target_pair"]
        entry = targets[key]
        original = read(entry["input_file"])
        canvas = headroom(original)
        human = mask_for(entry, key.split("-")[0])
        neutral = target_regions(original, human)["neutral_proxy"] & human
        references = pair(BASE / "bg-real-pixel-inputs", case["input"])
        previous = frozen[key]["subject"]
        panel_path = subject_dir / f"{key}.png"
        if digest(panel_path) != frozen[key]["panel_sha256"]:
            raise ValueError("Frozen subject panel changed")
        with Image.open(panel_path) as image:
            stored = np.asarray(image.convert("RGB"))

        baseline, candidate, fits = {}, {}, {}
        for role_index, (role, reference) in enumerate(references.items()):
            baseline[role], candidate[role], fits[role] = {}, {}, {}
            for col, (style, operation) in enumerate(EDITS.items()):
                tint, contrast = previous["estimated"][role][style]
                control = grade(canvas, (tint, contrast), .8)
                x, y = 8 + col * 264, 26 + (3 + role_index) * 280
                np.testing.assert_array_equal(control, stored[y:y + 256, x:x + 256])
                observed = grade(reference, operation)
                estimate, detail = contrast_from_pair(reference, observed)
                baseline[role][style] = control
                candidate[role][style] = grade(canvas, (tint, estimate), .8)
                fits[role][style] = {**detail, "original_contrast": contrast, "paired_contrast": estimate}

        dest = output / key
        dest.mkdir()
        file_hashes = {"subject": write_outputs(dest / "subject", baseline),
                       "paired": write_outputs(dest / "paired", candidate)}
        truth = {style: grade(canvas, coeffs, .8) for style, coeffs in EDITS.items()}
        sheet = {"prescribed edit": truth}
        sheet.update({f"{arm} / {role}": frames for arm, data in (("subject", baseline), ("paired", candidate))
                      for role, frames in data.items()})
        panel(dest / "comparison.png", sheet)
        base_scores = scores(baseline, truth, human, neutral)
        new_scores = scores(candidate, truth, human, neutral)
        increases = [new_scores["styles"][style]["response"][role]["target_mae"]
                     - base_scores["styles"][style]["response"][role]["target_mae"]
                     for role in references for style in EDITS]
        worst = max(worst, max(increases))
        records[key] = {"subject": base_scores, "paired": new_scores, "fits": fits, "output_sha256": file_hashes,
                        "worst_regression": max(increases), "panel_sha256": digest(dest / "comparison.png")}
    totals = {arm: sum(row[arm]["passed"] for row in records.values()) for arm in ("subject", "paired")}
    save_report(output / "results.json", {"subject_sha256": SUBJECT_SHA, "cases": records, "totals": totals,
                "worst_regression": worst, "passed": totals["paired"] >= 3 and worst <= 2.})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
