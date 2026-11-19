"""Test adjacent-bin smoothing on the frozen 13 reference chroma input."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints, verified
from probe_arri_shared_look import metrics
from probe_arri_two_looks import ASSIGNMENTS, CHECKPOINT
from probe_bg_real_editor_look import BASE, PARENT, PARENT_SHA, ROOT, STATES, evaluate, mask_for, save_panel
from probe_bg_real_native_reference import PARENT_RESULT_SHA, ROLES, verified_image
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from training_checks import finite_output


FILM_SOURCE_SHA = "e09286501decd0d41cf727bce645d9920298da166f6bdaf4bab20385d27d730e"
FILM_BASELINE_SHA = "fea85f0672c6da99d5734e8fb52034438122a03a71b7e493534c4eb38da07237"


def smooth(histogram):
    chroma = histogram[:, 1:]
    left = torch.cat((chroma[..., :1], chroma[..., :-1]), dim=-1)
    right = torch.cat((chroma[..., 1:], chroma[..., -1:]), dim=-1)
    shifted = .25 * left + .5 * chroma + .25 * right
    torch.testing.assert_close(chroma.sum(-1), shifted.sum(-1), atol=1e-6, rtol=0)
    return torch.cat((histogram[:, :1], shifted), dim=1)


def render(engine, target, reference, baseline):
    control = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
    np.testing.assert_array_equal(control, baseline)
    content, guide = engine.analysis(target), engine.analysis(reference)
    histogram, convert = (engine.api[key] for key in ("batch_histogram_1D", "srgb_to_ycbcr"))
    original = histogram(convert(guide))
    changed = smooth(original)
    np.testing.assert_array_equal(original[:, :1].cpu().numpy(), changed[:, :1].cpu().numpy())
    low = engine.api["TF"].resize(content, (64, 64), antialias=True)
    grid = engine.api["lowpass"](engine.model.apply_grid(low, torch.cat((histogram(convert(content)), changed), dim=1)))
    finite_output(grid, (1, 13, 8, 16, 16))
    output = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8)
    return output, (.5 * (original - changed).abs().sum(dim=-1))[0].tolist()


def save(path, image):
    from probe_arri_shared_look import digest
    Image.fromarray(image).save(path)
    return digest(path)


def mae(first, second):
    return float(np.abs(first.astype(np.int16) - second).mean())


def background(engine, out):
    evidence = verified_json(ROOT / "docs/research/2026-11-19-bg-real-editor-look-evidence.json", PARENT_RESULT_SHA)
    cases = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    folder = BASE / "bg-real-editor-look"
    results = {}
    for name, record in evidence["cases"].items():
        identity = record["artifacts"]
        target = verified_image(folder / name, "target.png", identity)
        out_case = out / name
        out_case.mkdir()
        output, previous, metadata = {}, {}, {}
        for role in ROLES:
            output[role], previous[role], metadata[role] = {}, {}, {}
            for state in STATES:
                key = f"{role}-{state}"
                reference = verified_image(folder / name, f"{key}-reference.png", identity)
                baseline = verified_image(folder / name, f"{key}-13.png", identity)
                candidate, shift = render(engine, target, reference, baseline)
                output[role][state], previous[role][state] = candidate, baseline
                metadata[role][state] = {"histogram_total_variation": shift, "change_from_13_rgb8": mae(candidate, baseline),
                                        "sha256": save(out_case / f"{key}-smooth.png", candidate)}
        truth = {"off": target, "on": verified_image(folder / name, "expected-on.png", identity)}
        mask = mask_for(cases[name], name.split("-")[0])
        result = evaluate(target, truth, output, mask)
        panel = out_case / "comparison.png"
        save_panel(panel, {"Expected": truth, "13 / original background": previous["C1"],
                           "13 / changed background": previous["F1"], "Smooth / original background": output["C1"],
                           "Smooth / changed background": output["F1"]})
        results[name] = result | {"artifacts": metadata, "baseline_luma_chroma": record["projection"],
                                  "baseline_swap_mae": record["swap_mae"]}
    return results


def film(engine, out):
    source = BASE / "arri-two-looks-v2"
    crossed = BASE / "arri-film-b-crossed-v1"
    earlier = verified_json(source / "results.json", FILM_SOURCE_SHA)
    baseline = verified_json(crossed / "results.json", FILM_BASELINE_SHA)
    images = {scene: endpoints(source, earlier, scene) for scene in ASSIGNMENTS}
    results = {}
    for name, row in baseline["cases"].items():
        target, style = images[row["target"]]
        predictions, metadata = [], {}
        for state, reference in zip(("off", "film_b"), images[row["reference"]], strict=True):
            tag = f"{row['target']}-from-{row['reference']}-{state}"
            old = verified(crossed / f"{tag}.png", row["output_sha256"][state])
            candidate, shift = render(engine, target, reference, old)
            predictions.append(candidate)
            metadata[state] = {"histogram_total_variation": shift, "change_from_13_rgb8": mae(candidate, old),
                               "sha256": save(out / f"{tag}-smooth.png", candidate)}
        expected = finish_spatial(target, style, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in (target, style)))
        values = {region: metrics(target, expected, *predictions, area)
                  for region, area in (("whole", whole), ("neutral", neutral))}
        results[name] = values | {"artifacts": metadata,
                                  "baseline": {region: row[region] for region in values}}
    return results


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13 checkpoint")
    out.mkdir()
    bg = background(engine, out)
    films = film(engine, out)
    gates = {
        "background": sum(row["passed"] for row in bg.values()) >= 3
        and max(row["max_protected_swap"] for row in bg.values()) <= 3,
        "film": all(row[region]["passed"] for name, row in films.items() if name.startswith("sparks_4592<-")
                    for region in ("whole", "neutral"))
        and all(row[region]["response_mae"] - row["baseline"][region]["response_mae"] <= 1.5
                for row in films.values() for region in ("whole", "neutral")),
        "unedited_references": all(row["artifacts"][role]["off"]["change_from_13_rgb8"] <= 1.5
                                   for row in bg.values() for role in ROLES)
        and all(row["artifacts"]["off"]["change_from_13_rgb8"] <= 1.5 for row in films.values()),
    }
    json_to(out / "results.json", {"model_sha256": CHECKPOINT, "background": bg, "film_b": films,
                                   "gates": gates, "passed": all(gates.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
