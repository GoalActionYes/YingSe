"""Offline separation test for matching colour distributions by corresponding regions."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from probe_bg_film_b_reference_path import (
    BASE, PARENT, PARENT_SHA, SOURCE, SOURCE_SHA, STRENGTH, checked_rgb,
    evaluate, finish_spatial, json_to, manifest, mask_for, panel, save, verified_json,
)
from probe_bg_real_crossed_grade_subject import core_mask
from probe_bg_real_crossed_grade_training import HELD_SHA, ROOT
from probe_sliced_color_distribution import linear


def inside(area):
    return np.asarray(Image.fromarray(area.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) == 255


def mapped(target, reference, from_region, to_region):
    if min(from_region.sum(), to_region.sum()) < 256:
        raise ValueError("Too little matching region")
    original = target.astype(np.float64) / 255
    guide = reference.astype(np.float64) / 255
    _, result = linear(original[from_region], guide[to_region], original)
    return result


def run(out):
    prior = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    held = verified_json(ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json", HELD_SHA)
    refs = {row["target_pair"]: row["input"] for row in held["cases"].values()}
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    if prior["qualified_cases"] != 4:
        raise ValueError("Unexpected qualification")
    out.mkdir()
    cases = {}
    for key, item in manifest(BASE, "bg-real-editor-look")["cases"].items():
        root = BASE / "bg-real-editor-look" / key
        target = checked_rgb(root / "target.png", item["artifacts"]["target.png"])
        person = mask_for(entries[key], key.split("-")[0])
        tm = [inside(person), inside(~person)]
        whole = np.ones(target.shape[:2], bool)
        bands = dict(person_core=tm[0], background_core=tm[1], boundary=~(tm[0] | tm[1]))
        opacity = np.asarray(Image.fromarray(person.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(2)),
                             dtype=np.float64)[..., None] / 255
        qualification = prior["qualification"][key]
        expected = checked_rgb(BASE / SOURCE / f"{key}-expected.png", qualification["expected_sha256"])
        truth = {"off": target, "on": expected}
        # Decode exactly the original subject alpha, then take conservative interiors.
        mask_entry = refs[key]["mask"]
        path = BASE / "bg-real-pixel-inputs" / mask_entry["name"]
        with Image.open(path) as im:
            ref_person = np.asarray(im.convert("L")) >= 128
        core = core_mask(path.parent, mask_entry, ref_person.shape)
        rm = [inside(ref_person) & core, inside(~ref_person)]
        arms = {a: {} for a in ("global", "region")}
        written, rows = {}, {}
        for role in ("C1", "F1"):
            name = f"{role}-off-reference.png"
            pair = {"off": checked_rgb(root / name, item["artifacts"][name]),
                    "on": checked_rgb(BASE / SOURCE / f"{key}-{role}-reference.png",
                            qualification["reference_diagnostics"][role]["edited_sha256"])}
            for arm in arms:
                arms[arm][role] = {}
            for state, ref in pair.items():
                if ref.shape[:2] != ref_person.shape:
                    raise ValueError("Unexpected reference geometry")
                full = mapped(target, ref, whole, np.ones_like(ref_person))
                front, back = (mapped(target, ref, a, b) for a,b in zip(tm, rm, strict=True))
                values = {"global": full, "region": opacity * front + (1 - opacity) * back}
                tag = f"{role}-{state}"
                written[tag] = {}
                rows[tag] = {"target": target, "reference": ref, "desired": truth[state]}
                for arm, image in values.items():
                    output = finish_spatial(target, np.rint(np.clip(image, 0, 1)*255).astype(np.uint8), STRENGTH)
                    arms[arm][role][state] = output
                    rows[tag][arm] = output
                    delta = np.abs(output.astype(np.int16) - truth[state])
                    written[tag][arm] = {
                        "sha256": save(out / f"{key}-{tag}-{arm}.png", output),
                        "mae": {k: float(delta[v].mean()) if v.any() else None for k,v in bands.items()},
                    }
        cases[key] = {"scores": {arm: evaluate(target, truth, data, person) for arm, data in arms.items()},
                      "supports": {"target": [int(v.sum()) for v in tm], "reference": [int(v.sum()) for v in rm]},
                      "outputs": written, "comparison_sha256": panel(out / f"{key}-comparison.png", rows)}
    summary = {}
    for arm in arms:
        readout = [c["scores"][arm] for c in cases.values()]
        count = sum(value["passed"] for value in readout)
        worst = max(value["max_protected_swap"] for value in readout)
        summary[arm] = {"passing_cases": count, "worst_protected_swap": worst, "passed": count >= 3 and worst <= 3}
    json_to(out / "results.json", {"cases": cases, "summary": summary,
                                   "source_sha256": SOURCE_SHA, "mask_evidence_sha256": HELD_SHA})
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
