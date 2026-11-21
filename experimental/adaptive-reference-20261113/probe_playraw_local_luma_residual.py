"""Test whether observed local luminance predicts a RAW-to-sRGB edit residual."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from probe_playraw_cross_editor_gain import PARENT_SHA, checked_image
from probe_playraw_mixed_light_cross_editor import FIT_SHA, PROTECTED, REGIONS_SHA, SKIN
from probe_playraw_mixed_light_transfer import LUMA, PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_playraw_pipeline_mask_attribution import MASK_SHA, SRGB_SHA, hash_pixels
from probe_playraw_rendered_input_operator import accepted
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import finish_spatial


def features(before):
    rgb = before.astype(np.float64) / 255
    red, green, blue = np.moveaxis(rgb, -1, 0)
    original_luma = rgb @ LUMA
    nearby_luma = cv2.GaussianBlur(original_luma, (0, 0), 2, borderType=cv2.BORDER_REFLECT_101)
    local = original_luma - nearby_luma
    quadratic = np.stack((np.ones_like(red), red, green, blue, red**2, green**2, blue**2,
                          red*green, red*blue, green*blue), axis=-1)
    return {"rgb": quadratic, "rgb_local_luma": np.concatenate((quadratic, local[..., None]), axis=-1)}


def train(before, raw_edit, srgb_edit, designs, parity):
    eligible = ((before >= 8) & (before <= 247)).all(axis=-1)
    residual = chroma(raw_edit) - chroma(srgb_edit)
    fits, ranks = {}, {}
    for arm, layout in designs.items():
        fits[arm], ranks[arm] = [], []
        for fold in (0, 1):
            use = eligible & (parity == fold)
            coefficients, _, rank, _ = np.linalg.lstsq(layout[use], residual[use], rcond=1e-5)
            if rank != layout.shape[-1] or not np.isfinite(coefficients).all():
                raise ValueError("Insufficient design rank")
            fits[arm].append(coefficients)
            ranks[arm].append(int(rank))
    return fits, ranks, [int(np.count_nonzero(eligible & (parity == fold))) for fold in (0, 1)]


def endpoint(folder, record, before):
    observed = {role: read(folder / f"{role}.tif", False)[0] for role in ("off", "on")}
    if {key: hash_pixels(value) for key, value in observed.items()} != record["rgb_sha"]:
        raise ValueError("Changed sRGB endpoints")
    np.testing.assert_array_equal(observed["off"], before)
    return observed["on"]


def evaluate(out, before, expected, srgb_edit, weight, zones, corrections, limits, prior):
    active = weight > 0
    base_chroma = chroma(before)
    target_delta = chroma(expected) - base_chroma
    srgb_delta = chroma(srgb_edit[active]) - base_chroma[active]
    luma = before[active] @ LUMA
    report, images = {}, {}
    for arm, offsets in {"base": 0., **corrections}.items():
        grade = before.copy()
        grade[active], gamut = render(base_chroma[active] + STRENGTH * weight[active, None] * (srgb_delta + offsets), luma)
        if arm == "base":
            np.testing.assert_array_equal(grade, prior)
        rows = {label: stats(chroma(grade) - base_chroma, target_delta, roi) for label, roi in zones.items()}
        clean = np.array_equal(grade[~active], before[~active])
        luma_max = float(np.abs(grade[active] @ LUMA - luma).max())
        passed = clean and luma_max <= .5 and all(
            accepted(row) if key not in limits else row[limits[key][0]] <= limits[key][1]
            for key, row in rows.items()
        )
        report[arm] = dict(scores=rows, luma_max=luma_max, exact_outside=clean, passed=passed,
                           gamut=gamut, sha=save(out / f"{arm}.png", grade))
        images[arm] = grade
    return dict(arms=report, board=panel(out / "comparison.png", {
        "known author operation": {"before": before, "expected": expected}, "renders": images,
    }))


def run(base, out):
    cv2.setNumThreads(1)
    folder = base / "playraw-disable-import-sharpen-v2"
    control = checked_json(folder / "results.json", SRGB_SHA)
    raw = base / "playraw-mask-response-v1"
    frozen = checked_json(raw / "results.json", MASK_SHA)
    author = base / "playraw-mixed-light-skin-v1/replay-v2"
    source = checked_json(author / "results.json", SOURCE_SHA)
    before, _ = read(author / "off.tif", False)
    after, mask = read(author / "on.tif", True)
    if [hash_pixels(frame) for frame in (before, after)] != source["rgb_sha256"] or hash_pixels(mask) != source["mask_pixels_sha256"]:
        raise ValueError("Changed local reference")
    raw_edit = checked_rgb(raw / "uniform-0.png", frozen["trials"][0]["image_sha256"])
    srgb_edit = endpoint(folder / "mixed", control["source"], before)
    old = checked_rgb(folder / "mixed/candidate.png", control["source"]["sha"])
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    parity = (y // 8 + x // 8) % 2
    designs = features(before)
    fits, ranks, samples = train(before, raw_edit, srgb_edit, designs, parity)
    active = mask > 0
    corrections = {}
    for arm, layout in designs.items():
        values = np.empty((int(active.sum()), 2))
        for fold in (0, 1):
            held = parity[active] != fold
            values[held] = layout[active][held] @ fits[arm][fold]
        corrections[arm] = values
    prior_fits = checked_json(base / "playraw-mixed-light-operator-v1/results.json", FIT_SHA)["cases"]
    zones = {f"{kind}-{fold}": (parity != fold) & part for fold in (0, 1)
             for kind, part in (("high", mask >= 204), ("transition", active & (mask < 204)))}
    limits = {f"transition-{fold}": ("response_mae", entry["arms"]["constant"]["scores"]["transition"]["response_mae"] + .25)
              for fold, entry in prior_fits.items()}
    out.mkdir()
    (out / "mixed").mkdir()
    cases = {"mixed": evaluate(out / "mixed", before, finish_spatial(before, after, STRENGTH), srgb_edit,
                               mask / 255, zones, corrections, limits, old)}

    target = base / "playraw-mixed-light-cross-editor-v2"
    info = checked_json(target / "results.json", PARENT_SHA)
    regions = checked_json(Path(__file__).with_name("darktable-cuckoo-legacy-regions.json"), REGIONS_SHA)
    for i, trial in enumerate(info["trials"]):
        before, expected = [checked_image(target / str(i), trial, name) for name in ("baseline", "editor-local")]
        stem = f"cuckoo-{i}"
        archive = folder / stem
        srgb_edit = endpoint(archive, control["targets"][i], before)
        old = checked_rgb(archive / "candidate.png", control["targets"][i]["sha"])
        y, x = np.ogrid[:before.shape[0], :before.shape[1]]
        zones = {name: (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                 for name, (x0, y0, x1, y1) in regions.items() if name in SKIN + PROTECTED}
        active = zones["cheek"] | zones["chest"]
        corrections = {arm: layout[active] @ np.mean(fits[arm], axis=0)
                       for arm, layout in features(before).items()}
        (out / stem).mkdir()
        cases[stem] = evaluate(out / stem, before, expected, srgb_edit, active.astype(float), zones, corrections,
                               {name: ("response_p95", 1.5) for name in PROTECTED}, old)
    protocol = Path(__file__).with_name("playraw-local-luma-residual-protocol.md")
    json_to(out / "results.json", dict(inputs=[SOURCE_SHA, MASK_SHA, SRGB_SHA, PARENT_SHA, REGIONS_SHA],
             ranks=ranks, samples=samples, coefficients={arm: np.array(tables).tolist() for arm, tables in fits.items()},
             protocol=digest(protocol), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
