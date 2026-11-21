"""Measure a darktable edit with a digital color chart and two lookup coordinates."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from probe_playraw_cross_editor_gain import PARENT_SHA, checked_image
from probe_playraw_mixed_light_cross_editor import FIT_SHA, REGIONS_SHA, SKIN, PROTECTED, default_states
from probe_playraw_mixed_light_transfer import LUMA, PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_playraw_rendered_input_operator import accepted, check_input, endpoints
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, digest, json_to, panel, save
from spatial_render import finish_spatial

PARENT = "8d430c18e5d90980cb334ec767029c542b50412c5399d54de7102dddaad5879c"


def chart(samples):
    tiles = samples.reshape(128, 256, 3)
    return tiles.repeat(4, axis=0).repeat(4, axis=1)


def palette(image):
    return image[2::4, 2::4].reshape(-1, 3)


def build_tables(base, out):
    cli = check_input(base, "cli")
    states = default_states(check_input(base, "author").read_bytes())
    rgb = np.rint(np.linspace(0, 255, 32)).astype(np.uint8)
    samples = np.stack(np.meshgrid(rgb, rgb, rgb, indexing="ij"), axis=-1).reshape(-1, 3)
    forward, receipt_a = endpoints(chart(samples), cli, states, out / "chart")
    reverse, receipt_b = endpoints(chart(samples[::-1]), cli, states, out / "chart-reordered")
    pixels = {state: palette(image) for state, image in forward.items()}
    checked = {state: int(np.abs(values.astype(np.int16) - palette(reverse[state][::-1, ::-1])).max())
               for state, values in pixels.items()}
    if max(checked.values()) > 1:
        raise ValueError(f"Chart measurements depend on tile order: {checked}")
    delta = chroma(pixels["on"]) - chroma(pixels["off"])
    trees = {arm: cKDTree(values) for arm, values in (("input", samples), ("closed_output", pixels["off"]))}
    facts = dict(forward=receipt_a, reordered=receipt_b, reordered_max=checked,
                 closed_colors=len(np.unique(pixels["off"], axis=0)), chart_colors=len(samples))
    return trees, delta, facts


def compare(out, before, expected, weight, zones, trees, delta, limits):
    output, metrics = {}, {}
    active = weight > 0
    luma = before[active] @ LUMA
    truth = chroma(expected) - chroma(before)
    for arm, tree in trees.items():
        distance, index = tree.query(before[active], k=4, workers=1)
        inverse = 1 / np.maximum(distance, .001)
        offsets = np.sum(delta[index] * inverse[..., None], axis=1) / inverse.sum(axis=1, keepdims=True)
        result = before.copy()
        result[active], gamut = render(chroma(before[active]) + STRENGTH * weight[active, None] * offsets, luma)
        scores = {name: stats(chroma(result) - chroma(before), truth, mask) for name, mask in zones.items()}
        luma_max = float(np.abs(result[active] @ LUMA - luma).max())
        clean = np.array_equal(result[~active], before[~active])
        passes = clean and luma_max <= .5 and all(
            accepted(value) if name not in limits else value[limits[name][0]] <= limits[name][1]
            for name, value in scores.items()
        )
        metrics[arm] = dict(scores=scores, sha=save(out / f"{arm}.png", result), passed=passes,
                            far_fraction=float((distance[:, 0] > 8).mean()), luma_max=luma_max,
                            exact_outside=clean, gamut=gamut)
        output[arm] = result
    return dict(arms=metrics, board=panel(out / "comparison.png", {
        "reference": {"before": before, "expected": expected}, "sampled response": output,
    }))


def source(base, out, trees, delta):
    archive = base / "playraw-mixed-light-skin-v1/replay-v2"
    frozen = checked_json(archive / "results.json", SOURCE_SHA)
    before, _ = read(archive / "off.tif", False)
    after, mask = read(archive / "on.tif", True)
    if [sha256(frame.tobytes()).hexdigest() for frame in (before, after)] != frozen["rgb_sha256"]:
        raise ValueError("Changed RGB input")
    if sha256(mask.tobytes()).hexdigest() != frozen["mask_pixels_sha256"]:
        raise ValueError("Changed mask")
    folds = checked_json(base / "playraw-mixed-light-operator-v1/results.json", FIT_SHA)["cases"]
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    parity = (y // 8 + x // 8) % 2
    zones = {f"{kind}-{fold}": (parity != fold) & part for fold in range(2)
             for kind, part in (("high", mask >= 204), ("transition", (mask > 0) & (mask < 204)))}
    limits = {f"transition-{fold}": ("response_mae",
              value["arms"]["constant"]["scores"]["transition"]["response_mae"] + .25)
              for fold, value in folds.items()}
    return compare(out, before, finish_spatial(before, after, STRENGTH), mask / 255, zones, trees, delta, limits)


def target(base, out, trial, frozen, rectangles, trees, delta):
    folder = base / "playraw-mixed-light-cross-editor-v2" / str(trial)
    before, expected = [checked_image(folder, frozen, tag) for tag in ("baseline", "editor-local")]
    zones = {}
    for name in SKIN + PROTECTED:
        x0, y0, x1, y1 = rectangles[name]
        mask = np.zeros(before.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = True
        zones[name] = mask
    weight = (zones["cheek"] | zones["chest"]).astype(float)
    return compare(out, before, expected, weight, zones, trees, delta,
                   {name: ("response_p95", 1.5) for name in PROTECTED})


def run(base, out):
    if any(not path.is_absolute() or not path.as_posix().isascii() for path in (base, out)):
        raise ValueError("Use absolute ASCII paths for the editor")
    checked_json(base / "playraw-rendered-input-operator-v1/results.json", PARENT)
    known = checked_json(base / "playraw-mixed-light-cross-editor-v2/results.json", PARENT_SHA)
    regions = checked_json(Path(__file__).with_name("darktable-cuckoo-legacy-regions.json"), REGIONS_SHA)
    out.mkdir()
    trees, delta, receipt = build_tables(base, out)
    cases = {}
    mixed = out / "mixed"
    mixed.mkdir()
    cases["mixed"] = source(base, mixed, trees, delta)
    for trial, frozen in enumerate(known["trials"]):
        dest = out / f"cuckoo-{trial}"
        dest.mkdir()
        cases[dest.name] = target(base, dest, trial, frozen, regions, trees, delta)
    protocol = Path(__file__).with_name("playraw-editor-chart-bridge-protocol.md")
    json_to(out / "results.json", dict(parent=PARENT, regions=REGIONS_SHA, protocol=digest(protocol),
                                       charts=receipt, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
