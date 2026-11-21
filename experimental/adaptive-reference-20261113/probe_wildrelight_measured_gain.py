"""Compare daylight reference normalization by measured exposure and RGB ratios."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_capture_invariant_look import render_style, style_transforms
from probe_gros_film_b_cross import score
from probe_gros_render_intent_projection import grade
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)
from probe_wildrelight_wall_cross import LUMA, SCENES, image, prepared_rgb

INPUT_SHA = "7ef5465396069bf7d4ba342d58cd713495d088c5fe81bdff09320f2fc0cf8503"
PARENT = "b0b75d2c38dd7b83736d917d0ba00597d4e2085d37cf70f4baf72d7f69be47c5"
REGIONS = "690db0ca2940354c985f99fa61be0485146eea08c2bcd005fce13d2d4340c0fc"


def gains(entry):
    earlier, later = (np.asarray(entry["envmaps"][time]["sphere_mean_rgb"]) for time in ("time0", "time5"))
    if not np.all(earlier > 0) or not np.all(later > 0):
        raise ValueError("Nonpositive environmental light estimate")
    return {"exposure": np.full(3, (earlier @ LUMA) / (later @ LUMA)), "rgb": earlier / later}


def areas_for(shape, rect):
    y, x = np.ogrid[:shape[0], :shape[1]]
    x0, y0, x1, y1 = rect
    roi = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
    if int(roi.sum()) != (x1 - x0) * (y1 - y0):
        raise ValueError("Changed score rectangle")
    return {"whole": np.ones(shape, dtype=bool), "neutral": roi}


def run(out):
    prepared = BASE / "wildrelight-wall-prepared-v1"
    raw = BASE / "wildrelight-small-v1"
    prior = BASE / "wildrelight-wall-cross-v1"
    source = checked_json(prepared / "inputs.json", INPUT_SHA)
    archive = checked_json(prior / "results.json", PARENT)
    labels = checked_json(Path(__file__).with_name("wildrelight-wall-cross-regions.json"), REGIONS)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or archive["strength"] != STRENGTH:
        raise ValueError("Changed renderer")
    film = style_transforms(BASE)["film_b"]
    out.mkdir()
    references, gains_report = {}, {}
    for scene in SCENES:
        record = source["scenes"][scene]
        initial, original = (checked_rgb(prepared / f"{scene}-{time}.png", record["hashes"][time])
                             for time in ("time0", "time5"))
        raw_name = f"small-aligned/{scene}/photo/time5_hdr.exr"
        if digest(raw / raw_name) != source["inputs"][raw_name]:
            raise ValueError("Unexpected late capture")
        pixels = image(raw / raw_name)
        np.testing.assert_array_equal(prepared_rgb(pixels, record["scale"]), original)
        field = areas_for(initial.shape[:2], labels[scene]["neutral"])
        by_arm = {"base": original} | {
            arm: prepared_rgb(pixels * coeff, record["scale"]) for arm, coeff in gains(record).items()
        }
        references[scene] = {arm: {"off": ref, "on": render_style(ref, film)} for arm, ref in by_arm.items()}
        gains_report[scene] = dict(gains={k: v.tolist() for k, v in gains(record).items()},
            differences={arm: {label: float(np.abs(initial[zone].astype(float) - ref[zone]).mean())
                               for label, zone in field.items()} for arm, ref in by_arm.items()},
            hashes={f"{arm}-{state}": save(out / f"{scene}-reference-{arm}-{state}.png", ref)
                    for arm, frames in references[scene].items() if arm != "base" for state, ref in frames.items()})

    cases = {}
    for scene, donor in (SCENES, SCENES[::-1]):
        records = archive["cases"][scene]
        target = checked_rgb(prepared / f"{scene}-time0.png", source["scenes"][scene]["hashes"]["time0"])
        expected = finish_spatial(target, render_style(target, film), STRENGTH)
        area = areas_for(target.shape[:2], labels[scene]["neutral"])
        base = {light: {state: checked_rgb(prior / f"{scene}-from-{donor}-{time}-{state}.png",
                                          records["hashes"][f"{time}-{state}"]) for state in ("off", "on")}
                for light, time in (("cold", "time0"), ("warm", "time5"))}
        if score(target, expected, base, area) != records["scores"]:
            raise ValueError("Unexpected prior scores")
        corrected = {arm: {state: grade(engine, target, ref) for state, ref in guide.items()}
                     for arm, guide in references[donor].items() if arm != "base"}
        scores = {"base": records["scores"]} | {
            arm: score(target, expected, {"cold": base["cold"], "warm": grades}, area) for arm, grades in corrected.items()
        }
        cases[scene] = dict(scores=scores, hashes={
            f"{arm}-{state}": save(out / f"{scene}-from-{donor}-{arm}-{state}.png", image)
            for arm, grades in corrected.items() for state, image in grades.items()
        }, board=panel(out / f"{scene}-comparison.png", {
            "target": {"original": target, "expected": expected}, "early reference result": base["cold"],
            "late baseline": base["warm"], **corrected,
        }))
    protocol = Path(__file__).with_name("wildrelight-measured-gain-protocol.md")
    json_to(out / "results.json", dict(parent=PARENT, input_sha=INPUT_SHA, regions=REGIONS,
                                      checkpoint=CHECKPOINT, strength=STRENGTH, protocol=digest(protocol),
                                      references=gains_report, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
