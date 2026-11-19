"""Stress the pinned cubic fit with separately captured near-matches."""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from probe_aligned_cubic_pilot import certify
from probe_arri_shared_look import metrics
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, STRENGTH
from probe_frozen_render_blend_bound import neutral
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_mls_film_light_cross import SOURCE_SHA
from probe_target_mirror_chroma import checked_rgb, panel, save
from reference_color_app.lut import apply_mapping
from spatial_render import SpatialEngine, finish_spatial


def load(folder, filename, sha):
    return checked_rgb(BASE / folder / filename, sha)


def exposure_pairs():
    folder = "dast-exposure-prepared-v1"
    record = verified_json(BASE / folder / "inputs.json",
                           "7e5137f4e559daf39e4f4f09a7500637beb897713257c4d7101de20a0d9dbf97")
    for group in record["groups"]:
        name = group["group"]
        frames = {state: load(folder, f"{name}/{state}.png", sha) for state, sha in group["frames"].items()}
        for state in ("repeat", "ue1", "ue2", "oe1", "oe2"):
            yield f"DAST-{name}-{state}", frames["nominal"], frames[state]


def paletted_cases():
    folder = "chromamill-capture-cross-v1"
    record = verified_json(BASE / folder / "results.json",
                           "ae4bdcfbaec10bf43c11b882138745ef02f86f58011d6102a43a666630e83789")
    for scene, values in record["prepared"].items():
        yield "CHROMA", scene, {name: tuple(load(folder, f"{scene}-{name}-{state}.png", v["hashes"][state])
                                           for state in ("off", "on")) for name, v in values.items()}, "white-213"
    folder = "mls-film-light-cross-v1"
    record = verified_json(BASE / folder / "results.json",
                           "add480d14eeee078b9108c4a1575719199637010bdacc92f21de0717ffe34754")
    originals = verified_json(BASE / "mls-linear-pilot/results.json", SOURCE_SHA)["scenes"]
    for scene in ("02", "03"):
        states = {}
        for light, suffix in (("standard", ""), ("blue", "_LED-B025")):
            filename = f"{scene}_2HAL_DESK{suffix}.png"
            states[light] = (load("mls-linear-pilot", filename, originals[scene]["inputs_sha256"][filename]),
                             load(folder, f"{scene}-{light}-reference-film.png", record["references"][scene][light]))
        yield "MLS", scene, states, "standard"
    folder = "hdm-showgirl-reference-prepared-v1"
    cases = verified_json(BASE / folder / "prepared.json",
                          "691c684dab8656b1e3090da0891aa1089013562bce648d40d70a1e001f8b872d")["cases"]
    for name, row in cases.items():
        pair = tuple(load(folder, f"{name}-{state}.png", row[state]["sha256"]) for state in ("original", "graded"))
        yield "Showgirl", name, {"frame": pair}, "frame"


class Stress:
    def __init__(self, out):
        self.out = out
        self.engine = None
        self.negatives = {}
        self.positives = {}

    def baseline(self, target, reference):
        if self.engine is None:
            self.engine = SpatialEngine("multiref-v1")
            if self.engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("13 checkpoint changed")
        return finish_spatial(target, self.engine.prepare(target, reference).apply(target), STRENGTH)

    def negative(self, name, target, reference):
        if target.shape != reference.shape or np.array_equal(target, reference):
            raise ValueError(f"Invalid separate-capture control: {name}")
        decision, table = certify(target, reference)
        row = {"decision": decision, "size": list(target.shape[:2]),
               "reference_difference_rgb8": float(np.abs(target.astype(np.int16) - reference).mean())}
        if table is not None:
            before = self.baseline(target, reference)
            after = finish_spatial(target, apply_mapping(target, table, 1), STRENGTH)
            row["vs_13_mae_rgb8"] = float(np.abs(after.astype(np.int16) - before).mean())
            region = neutral(target)
            if region.sum() >= 256:
                row["target_neutral_change_rgb8"] = {arm: float(np.abs(v[region].astype(np.int16) - target[region]).mean())
                                                     for arm, v in (("13", before), ("cubic", after))}
            row["output_sha256"] = {key: save(self.out / f"{name}-{key}.png", rgb)
                                    for key, rgb in (("13", before), ("cubic", after))}
            row["comparison_sha256"] = panel(self.out / f"{name}-comparison.png", {
                "separate captures": {"target": target, "reference": reference, "13": before, "cubic": after}})
        self.negatives[name] = row

    def positive(self, name, target, reference):
        decision, table = certify(target, reference)
        row = {"decision": decision}
        if table is not None:
            expected = finish_spatial(target, reference, STRENGTH)
            result = finish_spatial(target, apply_mapping(target, table, 1), STRENGTH)
            selected = neutral(target) & neutral(reference)
            row["whole"] = metrics(target, expected, target, result, np.ones(selected.shape, dtype=bool))
            row["neutral"] = metrics(target, expected, target, result, selected) if selected.sum() >= 256 else None
            row["output_sha256"] = save(self.out / f"{name}-positive.png", result)
        self.positives[name] = row


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    out.mkdir()
    trial = Stress(out)
    for key, target, reference in exposure_pairs():
        trial.negative(key, target, reference)
    anchors = {}
    for family, scene, states, original in paletted_cases():
        target = states[original][0]
        anchors.setdefault(family, {})[scene] = target
        for state, (raw, graded) in states.items():
            label = f"{family}-{scene}-{state}"
            trial.positive(label, raw, graded)
            if state != original:
                trial.negative(f"{label}-to-anchor", raw, target)
                trial.negative(f"{label}-from-anchor", target, raw)
    for family, group in anchors.items():
        for a, b in permutations(group, 2):
            trial.negative(f"{family}-{a}-from-{b}", group[a], group[b])
    if (len(trial.negatives), len(trial.positives)) != (47, 14):
        raise ValueError("Changed preregistered case counts")
    errors = sum(row["decision"]["route"] == "cubic" for row in trial.negatives.values())
    json_to(out / "results.json", {"13_sha256": CHECKPOINT, "strength": STRENGTH,
                                   "negatives": trial.negatives, "positives": trial.positives,
                                   "new_negative_activations": errors, "passed": errors == 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
