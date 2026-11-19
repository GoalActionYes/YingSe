"""Find the effect of overlapping target crops on fixed skin correction."""
import argparse
import itertools
from pathlib import Path
import re

import numpy as np

from probe_skin_spectral_preview_gate import verified, write_json
from probe_skin_spectral_scene_gate import REGIONS, STATES, image, ledger, save

INPUTS = "4de043a3e8dfb1f7202757609b4f7d5dcac092208723e55dc319dedf9cce4258"
MASKS = "f612caa372ef1090d2e07bd475b69cafe3b231509afd435102ec60ed991f84e7"
PARENT = "869c73766ef0f94d513959beada4c4ba31f2edb574ec1a9ad293e813e6e7d5c4"


def crops(rgb, model, canvas):
    height, width = rgb.shape[:2]
    half_height, half_width = (height + 1) // 2, (width + 1) // 2
    py, px = half_height // 8, half_width // 8
    for top, left in itertools.product((0, height // 4, height - half_height), (0, width // 4, width - half_width)):
        bottom, right = top + half_height, left + half_width
        confidence, _ = model.predict(rgb[top:bottom, left:right])
        y0, y1 = (py if top else 0), (half_height - py if bottom < height else half_height)
        x0, x1 = (px if left else 0), (half_width - px if right < width else half_width)
        canvas[top+y0:top+y1, left+x0:left+x1] |= confidence[y0:y1, x0:x1, 3] >= .70
    return canvas


def locate(base, output):
    from segment import SHA256, Segmenter

    inputs = base / "skin-spectral-scene-inputs-v1"
    masks = base / "skin-spectral-scene-masks-v1"
    frozen = ledger(masks / "masks.json", MASKS)
    if frozen["model"] != SHA256:
        raise ValueError("Unexpected locator")
    output.mkdir()
    result = {}
    model = Segmenter()
    try:
        for key, hashes in ledger(inputs / "inputs.json", INPUTS).items():
            result[key] = {}
            for state in STATES:
                name = f"{key}-{state}"
                rgb = image(inputs / f"{name}.png", hashes[state])
                baseline = image(masks / f"{name}.png", frozen["images"][key][state], "L") == 255
                combined = crops(rgb, model, baseline.copy())
                result[key][state] = save(output, name, combined.astype(np.uint8) * 255)
    finally:
        model.close()
    write_json(output / "masks.json", {"model": SHA256, "parent_sha256": MASKS, "images": result})


def render(base, masks, output):
    from probe_dast_local_edit_cross import INPUT_SHA
    from probe_skin_spectral_scene_gate import calibration, training
    from probe_skin_spectrum_rgb_bound import render as correct
    from segment import SHA256

    inputs = base / "skin-spectral-scene-inputs-v1"
    old_masks = base / "skin-spectral-scene-masks-v1"
    archive = base / "skin-spectral-scene-gate-v1"
    old = ledger(archive / "results.json", PARENT)["cases"]
    frozen = ledger(old_masks / "masks.json", MASKS)["images"]
    combined = ledger(masks / "masks.json")
    if combined["model"] != SHA256 or combined["parent_sha256"] != MASKS:
        raise ValueError("Unexpected locator or masks")
    fit, expected = training(base / "skin-spectra-input", base / "hyperspectral-faces-input")
    matrix = calibration()["A-D65"]
    residual = expected - fit @ matrix
    dast = base / "dast-exposure-prepared-v1"
    entries = ledger(dast / "inputs.json", INPUT_SHA)["groups"]
    known = {f"dast-{row['group']}": image(dast / str(row["group"]) / "nominal-mask.png",
                                            row["target_mask_sha256"], "L") > 0 for row in entries}
    locations = verified(Path(__file__).with_name("bg-real-skin-regions-protocol.md"), REGIONS)
    marks = re.findall(r"^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$",
                       locations.read_text(encoding="utf-8"), re.MULTILINE)
    protected = {"雪地", "浅墙", "白球服"}
    direct = {"面部小块", "前额", "后方人物脸颊"}
    screen = "投影幕中的脸"
    output.mkdir()
    result, tests, screen_checks = {}, [], []
    for key, hashes in ledger(inputs / "inputs.json", INPUTS).items():
        states, gates = {}, []
        for state in STATES:
            name = f"{key}-{state}"
            previous = old[key]["states"][state]
            rgb = image(inputs / f"{name}.png", hashes[state])
            was = image(old_masks / f"{name}.png", frozen[key][state], "L") == 255
            now = image(masks / f"{name}.png", combined["images"][key][state], "L") == 255
            added = now & ~was
            if np.any(was & ~now):
                raise ValueError("Crop union removed prior pixels")
            baseline = image(archive / f"{name}-matrix.png", previous["baseline_sha256"])
            final = image(archive / f"{name}-auto.png", previous["candidate_sha256"])
            if added.any():
                final[added] = correct(rgb[added], matrix, fit, residual)[0]
            delta = np.abs(final.astype(np.int16) - baseline)
            regions = {}
            for first, second, label, *bounds in marks:
                if key != f"{first}-{second}":
                    continue
                left, top, right, bottom = map(int, bounds)
                tile = np.s_[top:bottom, left:right]
                count = float(now[tile].mean())
                peak = int(delta[tile].max())
                error = float(delta[tile].mean())
                regions[label] = {"selected": count, "peak": peak, "rgb_mae": error}
                if label in direct:
                    tests.append(count >= .95)
                if label in protected:
                    tests.append(count == 0 and peak == 0)
                if label == screen:
                    before = previous["areas"][label]
                    screen_checks.append(count <= before["selected"] and error <= before["rgb_mae"])
            if key in known:
                count = float(now[known[key]].mean())
                tests.append(count >= .95)
                regions["archived_core"] = {"selected": count}
            gates.append(now)
            states[state] = {"added_fraction": float(added.mean()), "areas": regions,
                             "sha256": save(output, f"{name}-auto", final)}
        first, second = gates
        union = int(np.count_nonzero(first | second))
        result[key] = {"states": states,
                       "intersection_over_union": float(np.count_nonzero(first & second) / union) if union else None}
    if (len(tests), len(screen_checks)) != (22, 2):
        raise ValueError("Missing checks")
    write_json(output / "results.json", {"parent_sha256": PARENT, "cases": result, "passing_checks": sum(tests),
                                         "screen_checks": screen_checks, "passed": all(tests) and all(screen_checks)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    for mode, args in (("locate", ("base", "output")), ("render", ("base", "masks", "output"))):
        child = tasks.add_parser(mode)
        for name in args:
            child.add_argument(name, type=Path)
    options = vars(parser.parse_args())
    {"locate": locate, "render": render}[options.pop("task")](**options)
