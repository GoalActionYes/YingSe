"""Compare lossless local RGB replay after verified same-capture correspondence."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
import torch

from background_intervention import target_regions
from probe_aligned_cubic_capture_stress import paletted_cases
from probe_aligned_cubic_fine_residual import recode
from probe_arri_shared_look import metrics
from probe_gros_local_editor_channel import actual_png, frozen_json, request, serving
from probe_gros_mixed_light_exposure_toggle import TRANSFER
from probe_gros_mixed_light_transfer import LEVELS
from probe_playraw_flash_room_workbench import SOURCE_RECEIPT, STATES, added_endpoints, evaluate, prepare
from probe_playraw_mixed_light_appearance import STORE, STORE_SHA, image_measure
from probe_playraw_mixed_light_geometric_replay import align
from probe_playraw_mixed_light_transfer import BASE, PARENT, PARENT_SHA, chroma, read
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench
from verify_development_inputs import manifest


FLASH_WORKBENCH = "d93ae02d3f5125cd8273fbe333797053fa86d45ac5352ea3c2fdcfd99501d390"


def shared_detail(target, placed, support):
    pixels = target.astype(np.int16)
    contrast_right = np.abs(pixels[:-1, 1:] - pixels[:-1, :-1]).max(axis=2)
    contrast_down = np.abs(pixels[1:, :-1] - pixels[:-1, :-1]).max(axis=2)
    eligible = support[:-1, :-1] & support[:-1, 1:] & support[1:, :-1]
    eligible &= np.maximum(contrast_right, contrast_down) >= 2
    exact = (target == placed).all(axis=2)
    shared = eligible & exact[:-1, :-1] & exact[:-1, 1:] & exact[1:, :-1]
    height, width = eligible.shape
    counts = []
    y_edges = np.linspace(0, height, 5, dtype=int)
    x_edges = np.linspace(0, width, 5, dtype=int)
    for ya, yb in zip(y_edges[:-1], y_edges[1:]):
        for xa, xb in zip(x_edges[:-1], x_edges[1:]):
            region = np.s_[ya:yb, xa:xb]
            counts.append((int(eligible[region].sum()), int(shared[region].sum())))
    qualifying = sum(n >= 128 and 2 * k >= n for n, k in counts)
    return dict(qualifying=qualifying, eligible=int(eligible.sum()), shared=int(shared.sum()),
                counts=counts, accepted=qualifying >= 8)


def proposals(sift, target, reference, fallback):
    placed, support, trace = align(sift, target, reference)
    integer = False
    detail = None
    if placed is not None:
        matrix = np.asarray(trace[-1]["matrix"])
        integer = np.array_equal(matrix[:, :2], np.eye(2)) and np.array_equal(matrix[:, 2], np.rint(matrix[:, 2]))
    output = fallback
    if integer:
        detail = shared_detail(target, placed, support)
        output = target.copy()
        output[support] = finish_spatial(target[support][None, :, :], placed[support][None, :, :], STRENGTH)[0]
        np.testing.assert_array_equal(output[~support], target[~support])
    use = integer and detail["accepted"]
    return dict(geometry=output, guarded=output if use else fallback), dict(
        alignment=trace, integer=integer, detail=detail, accepted=use)


def verify_detail():
    height, width = 128, 128
    pixels = np.random.default_rng(274).integers(32, 224, (height, width, 3), dtype=np.uint8)
    support = np.ones((height, width), bool)
    assert shared_detail(pixels, pixels, support)["qualifying"] == 16
    altered = pixels.copy()
    altered[:, width // 2:] += 9
    assert shared_detail(pixels, altered, support)["qualifying"] == 8
    assert not shared_detail(pixels, pixels + 1, support)["accepted"]
    flat = np.full_like(pixels, 128)
    assert not shared_detail(flat, flat, support)["accepted"]


def positives(output):
    flash_source = frozen_json(Path(__file__).resolve().parents[2] / "docs/research/2026-11-24-playraw-flash-room-evidence.json",
                               SOURCE_RECEIPT)
    flash_archive = BASE / "playraw-flash-room-workbench-v1"
    flash_report = checked_json(flash_archive / "results.json", FLASH_WORKBENCH)
    temp = output / "reference-crops"
    temp.mkdir()
    _, crops, frames, boxes = prepare(BASE / "playraw-flash-room-rawtherapee-v1", temp, flash_source)
    flash = frames["off"]
    flash_expected = finish_spatial(flash, frames["on"], STRENGTH)

    evidence = checked_json(PARENT / "results.json", PARENT_SHA)
    history = checked_json(STORE / "results.json", STORE_SHA)
    portrait, _ = read(PARENT / "off.tif", False)
    edited, mask = read(PARENT / "on.tif", True)
    identities = [sha256(image.tobytes()).hexdigest() for image in (portrait, edited)]
    if identities != evidence["rgb_sha256"] or sha256(mask.tobytes()).hexdigest() != evidence["mask_pixels_sha256"]:
        raise ValueError("Existing author samples changed")
    height, width = mask.shape
    crop = np.s_[height // 16:height - height // 16, width // 16:width - width // 16]
    portrait_expected = finish_spatial(portrait, edited, STRENGTH)
    zones = {"author": mask >= 204, "protected": mask == 0,
             "neutral": target_regions(portrait, np.zeros_like(mask, bool))["neutral_proxy"] & (mask == 0)}

    for view in ("full", "crop"):
        refs = frames if view == "full" else {state: checked_rgb(crops[state], digest(crops[state])) for state in STATES}
        archived = flash_report["scores"]["spatial13"][view]["output_sha256"]
        fallback = {s: checked_rgb(flash_archive / f"spatial13-{view}-{s}.png", archived[s]) for s in STATES}
        yield f"flash-{view}", flash, refs, fallback, flash_expected, boxes

        area = np.s_[:] if view == "full" else crop
        fallback = {state: checked_rgb(STORE / f"{view}-{state}.png",
                   history["cases"][view]["outputs"][state]["image_sha256"]) for state in STATES}
        yield f"skin-{view}", portrait, {"off": portrait[area], "on": edited[area]}, fallback, portrait_expected, zones


def score(name, target, expected, frames, zones):
    off, on = (frames[state] for state in STATES)
    if name.startswith("flash"):
        return evaluate(target, expected, off, on, zones)
    checks = image_measure([off, on], chroma(expected) - chroma(target), zones, False)
    rgb = metrics(target, expected, off, on, zones["author"])
    clipped = {state: added_endpoints(target, frame) for state, frame in frames.items()}
    return dict(chroma=checks, rgb_author=rgb, added_endpoints=clipped,
                passed=checks["passed"] and rgb["passed"] and rgb["on_matches_authored_mae"] <= 3
                and max(clipped.values()) <= .02)


def positive_trials(sift, output):
    rows = {}
    for name, target, references, baselines, expected, zones in positives(output):
        by_arm = {"baseline": baselines, "geometry": {}, "guarded": {}}
        decisions = {}
        for state, reference in references.items():
            outcomes, decisions[state] = proposals(sift, target, reference, baselines[state])
            for arm, frame in outcomes.items():
                by_arm[arm][state] = frame
        scores = {}
        board = {"inputs": {"target": target, "intended": expected}}
        for arm, outputs in by_arm.items():
            files = {state: save(output / f"{name}-{arm}-{state}.png", frame) for state, frame in outputs.items()}
            scores[arm] = dict(score(name, target, expected, outputs, zones), hashes=files,
                               off_exact=bool(np.array_equal(outputs["off"], target)))
            board[arm] = outputs
        passed = all(d["accepted"] for d in decisions.values())
        passed = passed and scores["guarded"]["passed"] and scores["guarded"]["off_exact"]
        rows[name] = dict(scores=scores, decisions=decisions, passed=passed,
                          panel=panel(output / f"{name}-comparison.png", board))
        print(name, "passed", passed, flush=True)
    return rows


def negatives():
    archived = BASE / "gros-mixed-light-transfer-repeat-v1"
    history = checked_json(archived / "results.json", TRANSFER)
    for subject, known in history["prepared_hashes"].items():
        files = {level: checked_rgb(archived / f"reference-{subject}-{level}.png", known[level]) for level in LEVELS}
        for first, second in permutations(LEVELS, 2):
            yield f"gros-{subject}-{first}-{second}", files[first], files[second]
    for family, name, states, _ in paletted_cases():
        if family == "CHROMA" and name in ("Scene1_a", "Scene2_a"):
            first, second = (("white-241", "white-213") if name == "Scene1_a" else ("white-213", "white-241"))
            yield f"chart-{name}", states[first][0], states[second][0]

    jersey = BASE / "bg-real-editor-look" / "872-1268"
    sources = manifest(BASE, "bg-real-editor-look")["cases"]["872-1268"]["artifacts"]
    target = checked_rgb(jersey / "target.png", sources["target.png"])
    source = BASE / "playraw-flash-room-rawtherapee-v1"
    receipt = frozen_json(Path(__file__).resolve().parents[2] / "docs/research/2026-11-24-playraw-flash-room-evidence.json",
                          SOURCE_RECEIPT)
    flash = {s: checked_rgb(source / s / "image.png", receipt["verification"]["png_file_hashes"][s]) for s in STATES}
    for state, reference in flash.items():
        yield f"jersey-{state}", target, reference
    yield "jpeg-flash", recode(flash["off"]), recode(flash["on"])


def negative_trials(sift, output):
    rows = {}
    with serving(make_spatial_workbench(output / "workbench", enhanced=True)) as server:
        if server.store.engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Unexpected baseline checkpoint")
        input_path = output / "input.png"
        for name, target, reference in negatives():
            save(input_path, target)
            t = request(server, "/api/media", input_path.read_bytes(), f"{name}-target.png")["id"]
            save(input_path, reference)
            r = request(server, "/api/media", input_path.read_bytes(), f"{name}-reference.png")["id"]
            base, base_sha = actual_png(server, t, r, output / f"{name}-baseline.png")
            outcomes, decision = proposals(sift, target, reference, base)
            scores = {arm: save(output / f"{name}-{arm}.png", frame) for arm, frame in outcomes.items()}
            same = bool(np.array_equal(base, outcomes["guarded"]))
            rows[name] = dict(decision, output_sha256={"baseline": base_sha, **scores}, fallback_exact=same)
            if name == "jpeg-flash" and decision["accepted"]:
                regions = Path(__file__).with_name("playraw-flash-room-regions.json")
                rectangles = frozen_json(regions, "388bf16164f4da39d04bdfe8a287056deaec095b385cc5f115246c3ff9350615")
                boxes = {key: np.s_[top:bottom, left:right]
                         for key, (left, top, right, bottom) in rectangles["rectangles"].items()}
                known = finish_spatial(target, reference, STRENGTH)
                rows[name]["score"] = evaluate(target, known, target, outcomes["guarded"], boxes)
            print(name, "integer", decision["integer"], "accepted", decision["accepted"], flush=True)
    return rows


def run(output):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    torch.set_num_threads(4)
    verify_detail()
    output.mkdir()
    sift = cv2.SIFT_create()
    accepted = positive_trials(sift, output)
    rejected = negative_trials(sift, output)
    jpeg = rejected["jpeg-flash"]
    jpeg_ok = jpeg["score"]["passed"] if jpeg["accepted"] else jpeg["fallback_exact"]
    passed = (len(accepted) == 4 and len(rejected) == 17 and all(row["passed"] for row in accepted.values())
              and all(not row["accepted"] and row["fallback_exact"] for name, row in rejected.items() if name != "jpeg-flash")
              and jpeg_ok)
    code = Path(__file__)
    json_to(output / "results.json", dict(checkpoint=CHECKPOINT, strength=STRENGTH, opencv=cv2.__version__,
        positive=accepted, negative=rejected, passed=passed, script_sha256=digest(code),
        protocol_sha256=digest(code.with_name("same-capture-detail-replay-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
