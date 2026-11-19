"""Offline leave-one-photo-family-out Film B appearance response replay."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_bg_film_b_reference_path import (
    BASE, CHECKPOINT, COUNTS, PARENT, PARENT_SHA, SOURCE, SOURCE_SHA, STRENGTH, WEIGHT, WEIGHT_SHA256,
    PreparedSpatial, SpatialEngine, appearance_grid, checked_rgb, digest, evaluate, finish_spatial,
    initialize_replication_model, json_to, manifest, mask_for, panel, save, verified_json, verify,
)

VARIANTS = ("C1", "F1")


def adapter_value(engine, content, reference):
    model, api = engine.model, engine.api
    feature = lambda image: model.img_projection(api["TF"].resize(image, (64, 64), antialias=True))
    return model.reference_adapter(feature(content), feature(reference))


def render(engine, target, content, stats, appearance, correction=None):
    adapter = engine.model.reference_adapter
    if correction is None:
        grid = appearance_grid(engine.model, engine.api, content, stats, appearance_reference=appearance)
    else:
        with adapter.register_forward_hook(lambda _module, _args, value: value + correction):
            grid = appearance_grid(engine.model, engine.api, content, stats, appearance_reference=appearance)
    return finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    prior = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    if prior["qualified_cases"] != 4 or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Unexpected qualification or checkpoint")
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected base checkpoint")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    rows = manifest(BASE, "bg-real-editor-look")["cases"]
    if len({part for key in rows for part in key.split("-")}) != 2 * len(rows):
        raise ValueError("Reference/target identifier overlap")
    sources = {}
    for key, row in rows.items():
        root = BASE / "bg-real-editor-look" / key
        q = prior["qualification"][key]
        target = checked_rgb(root / "target.png", row["artifacts"]["target.png"])
        refs = {}
        for v in VARIANTS:
            name = f"{v}-off-reference.png"
            before = checked_rgb(root / name, row["artifacts"][name])
            after = checked_rgb(BASE / SOURCE / f"{key}-{v}-reference.png",
                                q["reference_diagnostics"][v]["edited_sha256"])
            refs[v] = tuple(engine.analysis(image) for image in (before, after))
        sources[key] = target, refs
    out.mkdir()
    cases = {}
    for key, (target, refs) in sources.items():
        content = engine.analysis(target)
        donor_keys = [other for other in sources if other != key]
        differences = [adapter_value(engine, content, on) - adapter_value(engine, content, off)
                       for donor in donor_keys for off, on in sources[donor][1].values()]
        average = torch.stack(differences).mean(dim=0)
        frozen = prior["comparison"]["expanded"]["cases"][key]
        histories = {arm: {} for arm in ("full", "statistics", "replay")}
        pictures, receipts = {}, {}
        for v, (off, on) in refs.items():
            rgb0, rgb1 = [checked_rgb(BASE / SOURCE / f"expanded-{key}-{v}-{state}.png",
                                     frozen["outputs"][f"{v}-{state}"]) for state in ("off", "on")]
            off_value, on_value = (adapter_value(engine, content, image) for image in (off, on))
            local = on_value - off_value
            np.testing.assert_array_equal(render(engine, target, content, on, on, local - local), rgb1)
            stat = render(engine, target, content, on, off)
            replay = render(engine, target, content, on, on, average - local)
            for arm, result in zip(histories, (rgb1, stat, replay), strict=True):
                histories[arm][v] = {"off": rgb0, "on": result}
            receipts[v] = save(out / f"{key}-{v}-replay.png", replay)
            pictures[v] = {"target": target, "expanded": rgb1, "statistics": stat, "donor replay": replay}

        truth = checked_rgb(BASE / SOURCE / f"{key}-expected.png", prior["qualification"][key]["expected_sha256"])
        mask = mask_for(entries[key], key.split("-")[0])
        stats = {arm: evaluate(target, {"off": target, "on": truth}, outputs, mask)
                 for arm, outputs in histories.items()}
        if stats["full"] != frozen["scores"]:
            raise ValueError(f"Frozen scores differ: {key}")
        for images in pictures.values():
            images["expected"] = truth
        cases[key] = {"scores": stats, "donors": donor_keys, "outputs": receipts,
                      "comparison_sha256": panel(out / f"{key}-comparison.png", pictures)}
    summaries = {}
    for arm in histories:
        scores = [record["scores"][arm] for record in cases.values()]
        count = sum(s["passed"] for s in scores)
        worst = max(s["max_protected_swap"] for s in scores)
        summaries[arm] = {"passing_cases": count, "worst_protected_swap": worst, "passed": count >= 3 and worst <= 3}
    json_to(out / "results.json", {"source_sha256": SOURCE_SHA, "weight_sha256": WEIGHT_SHA256,
                                   "strength": STRENGTH, "cases": cases, "summaries": summaries})
    print(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
