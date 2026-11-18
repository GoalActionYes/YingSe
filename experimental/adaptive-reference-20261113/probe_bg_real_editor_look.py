"""Measure a frozen HSL and RGB-curve look across the four BG-REAL swaps."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from background_intervention import LUMA, target_regions
from probe_bg_real_crossed_grade_training import BASE, HELD_SHA, PARENT, PARENT_SHA, ROOT, digest, mask_for, read
from probe_reference_chroma_spread import projection
from reference_color_app.adjustments import transform
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial
from spatial_store import SpatialStore

EDIT = {
    "hsl": {"blue": [-35, 35, -12], "cyan": [-20, 20, -6], "green": [15, -15, 0]},
    "curves": {"rgb": [[0, 0], [.25, .20], [.5, .5], [.75, .80], [1, 1]]},
}
STRENGTH = .8
STATES = ("off", "on")


def authored(image):
    values = transform(image.astype(np.float32) / 255, EDIT)
    return np.rint(values * 255).astype(np.uint8)


def split_signal(after, before):
    delta = after.astype(np.float64) - before
    y = delta @ LUMA
    chroma = np.stack([(delta[..., 2] - y) / (2 * (1 - LUMA[2])),
                       (delta[..., 0] - y) / (2 * (1 - LUMA[0]))], axis=-1)
    return {"luma": y, "chroma": chroma}


def save_panel(path, rows):
    canvas = Image.new("RGB", (2 * 272, len(rows) * 282), "#222730")
    draw = ImageDraw.Draw(canvas)
    for i, (label, values) in enumerate(rows.items()):
        for j, (state, pixels) in enumerate(values.items()):
            x, y = j*272 + 8, i*282
            draw.text((x, y+4), f"{label} - {state}", fill="white")
            tile = ImageOps.contain(Image.fromarray(pixels), (256, 256))
            canvas.paste(tile, (x + (256-tile.width)//2, y + 22 + (256-tile.height)//2))
    canvas.save(path)


def evaluate(target, truth, renders, person):
    regions = target_regions(target, person)
    protected = {"person": person,
                 "neutral": regions["neutral_proxy"],
                 "neutral_person": regions["neutral_proxy"] & person}
    desired = split_signal(truth["on"], truth["off"])
    strength = {component: float(np.abs(values).mean()) for component, values in desired.items()}
    responses = {}
    for role, values in renders.items():
        actual = split_signal(values["on"], values["off"])
        responses[role] = {
            name: projection(actual[name], delta, regions["whole"]) if strength[name] >= 1 else None
            for name, delta in desired.items()
        }
    swaps = {}
    for state in STATES:
        delta = np.abs(renders["C1"][state].astype(np.int16) - renders["F1"][state])
        swaps[state] = {name: float(delta[mask].mean()) if mask.sum() >= 256 else None
                        for name, mask in protected.items()}
    stability = [x for row in swaps.values() for x in row.values() if x is not None]
    return dict(
        expected_mean_abs_delta=strength, projection=responses, swap_mae=swaps,
        region_pixels={name: int(mask.sum()) for name, mask in protected.items()},
        passed=bool(person.sum() >= 256 and protected["neutral"].sum() >= 256
                    and all(x is not None and .75 <= x <= 1.25 for row in responses.values() for x in row.values())
                    and max(stability) <= 1.5),
        max_protected_swap=max(stability),
    )


def run(output):
    torch.set_num_threads(4)
    held = ROOT/"docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    parent = PARENT/"results.json"
    if (digest(held), digest(parent)) != (HELD_SHA, PARENT_SHA):
        raise ValueError("Frozen input groups changed")
    registered = json.loads(parent.read_bytes())
    engine = SpatialEngine("multiref-v1")
    if registered["model_sha256"] != engine.checkpoint_sha or registered["strength"] != STRENGTH:
        raise ValueError("Unexpected model or strength")
    targets = {row["pair"]: row for row in registered["cases"]}
    cases = list(json.loads(held.read_bytes())["cases"].values())
    if len(cases) != 4:
        raise ValueError("Expected four frozen cases")
    output.mkdir()
    report = {}
    for case in cases:
        key = case["target_pair"]
        entry = targets[key]
        folder = output/key
        folder.mkdir()
        target = read(entry["input_file"])
        store = SpatialStore(folder/"store", engine)

        def save(name, pixels):
            file = folder/f"{name}.png"
            Image.fromarray(pixels).save(file)
            return file

        target_media = store.add_media(save("target", target))
        references, rendered = {}, {}
        for role in ("C1", "F1"):
            item = case["input"][role]
            path = BASE/"bg-real-pixel-inputs"/item["name"]
            if digest(path) != item["sha256"]:
                raise ValueError("Frozen reference changed")
            photo = decode_image(path)
            references[role] = {"off": photo, "on": authored(photo)}
            rendered[role] = {}
            for state, pixels in references[role].items():
                media = store.add_media(save(f"{role}-{state}-reference", pixels))
                mapping = store.analyze(target_media["id"], media["id"])
                if mapping["diagnostics"]["certified_aligned"]["accepted"]:
                    raise ValueError("Unexpected same-content path for cross-scene comparison")
                result = store.render(mapping["id"], target, STRENGTH)
                rendered[role][state] = result
                save(f"{role}-{state}-13", result)

        truth = {"off": target, "on": finish_spatial(target, authored(target), STRENGTH)}
        save("expected-on", truth["on"])
        person = mask_for(entry, key.split("-")[0])
        save_panel(folder/"comparison.png", {"expected target": truth, **{
            f"{role} reference": references[role] for role in references}, **{
            f"{role} output 13": rendered[role] for role in rendered}})
        report[key] = dict(evaluate(target, truth, rendered, person),
                           artifacts={p.name: digest(p) for p in sorted(folder.glob("*.png"))})

    passed = sum(row["passed"] for row in report.values())
    worst = max(row["max_protected_swap"] for row in report.values())
    result = {"model_sha256": engine.checkpoint_sha, "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("bg-real-editor-look-protocol.md")),
              "edit": EDIT, "strength": STRENGTH, "cases": report, "passing_cases": passed,
              "max_protected_swap": worst, "passed": passed >= 3 and worst <= 3}
    (output/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
