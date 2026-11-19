"""Cross the two frozen reference paths on Film B background-swap photos."""

import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from verify_development_inputs import manifest, verify


SOURCE = "bg-real-film-b-qualification-v1"
SOURCE_SHA = "881bf466bd4456ba5b6a4ffe5a3126ddb7fb3de25912291a5790b9fb83813e16"
ARMS = ("full", "statistics", "appearance")


def render(engine, target, reference_pair):
    content = engine.analysis(target)
    off, on = [engine.analysis(image) for image in reference_pair]
    results = {}
    inputs = {"off": (off, off), "full": (on, on), "statistics": (on, off), "appearance": (off, on)}
    for mode, (stats, appearance) in inputs.items():
        grid = appearance_grid(engine.model, engine.api, content, stats, appearance_reference=appearance)
        results[mode] = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)
    return results


def compare(engine, out, key, prior, raw, entry):
    root = BASE / "bg-real-editor-look" / key
    source = BASE / SOURCE
    qualified = prior["qualification"][key]
    frozen = prior["comparison"]["expanded"]["cases"][key]
    original = checked_rgb(root / "target.png", raw["target.png"])
    styled = checked_rgb(source / f"{key}-expected.png", qualified["expected_sha256"])
    expected = {"off": original, "on": styled}
    person = mask_for(entry, key.split("-")[0])
    films = {arm: {} for arm in ARMS}
    hashes, sheet = {}, {}
    for variant in ("C1", "F1"):
        path = f"{variant}-off-reference.png"
        references = (
            checked_rgb(root / path, raw[path]),
            checked_rgb(source / f"{key}-{variant}-reference.png",
                        qualified["reference_diagnostics"][variant]["edited_sha256"]),
        )
        results = render(engine, original, references)
        for state, mode in (("off", "off"), ("on", "full")):
            tag = f"{variant}-{state}"
            archive = checked_rgb(source / f"expanded-{key}-{tag}.png", frozen["outputs"][tag])
            np.testing.assert_array_equal(results[mode], archive)
        sheet[variant] = {"target": original, "known edit": styled, **{arm: results[arm] for arm in ARMS}}
        for arm in ARMS:
            films[arm][variant] = {"off": results["off"], "on": results[arm]}
        hashes[variant] = {arm: save(out / f"{key}-{variant}-{arm}.png", results[arm]) for arm in ARMS[1:]}
    scores = {arm: evaluate(original, expected, images, person) for arm, images in films.items()}
    if scores["full"] != frozen["scores"]:
        raise ValueError(f"Frozen full scores changed: {key}")
    interaction = {
        variant: {tone: scores["full"]["projection"][variant][tone]
                        - sum(scores[arm]["projection"][variant][tone] for arm in ARMS[1:])
                  for tone in ("luma", "chroma")}
        for variant in ("C1", "F1")
    }
    return {"scores": scores, "interaction_projection": interaction, "output_sha256": hashes,
            "comparison_sha256": panel(out / f"{key}-comparison.png", sheet)}


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    prior = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    if prior["qualified_cases"] != 4 or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Frozen test qualification or candidate changed")
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    cases = {key: compare(engine, out, key, prior, row["artifacts"], entries[key])
             for key, row in manifest(BASE, "bg-real-editor-look")["cases"].items()}
    summaries = {}
    for arm in ARMS:
        scores = [row["scores"][arm] for row in cases.values()]
        count = sum(row["passed"] for row in scores)
        worst = max(row["max_protected_swap"] for row in scores)
        summaries[arm] = {"passing_cases": count, "max_protected_swap": worst, "passed": count >= 3 and worst <= 3}
    json_to(out / "results.json", {"source_sha256": SOURCE_SHA, "checkpoint_sha256": WEIGHT_SHA256,
                                   "strength": STRENGTH, "cases": cases, "summaries": summaries})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
