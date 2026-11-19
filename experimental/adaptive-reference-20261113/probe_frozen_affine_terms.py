"""Cross the normalized affine matrix and bias of frozen reference states."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_arri_film_b_dose import endpoints
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256
from probe_frozen_internal_clamp import byte_rgb
from probe_frozen_render_blend_bound import FULL, FULL_SHA, Y, neutral
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb, save
from spatial_render import SpatialEngine, coordinates, finish_spatial
from verify_development_inputs import manifest, verify

STATES = ("off", "matrix", "bias", "on")


def render_states(engine, target, references):
    model = engine.model
    analysis = engine.analysis(target)
    grids = [appearance_grid(model, engine.api, analysis, engine.analysis(ref)) for ref in references]
    images = {state: np.empty_like(target) for state in STATES}
    height, width = target.shape[:2]
    weights = F.softmax(model.sampler.L_weights, dim=1)
    for start in range(0, height, 256):
        stop = min(start + 256, height)
        content = engine.tensor(target[start:stop])
        depth = (model.sampler.pre_guide(content) * weights).sum(1)
        coord = coordinates(width, height, start, stop, depth)
        off, on = (F.grid_sample(g, coord, mode="bilinear", padding_mode="border",
                                 align_corners=False).squeeze(2) for g in grids)
        if torch.any(off[:, -1:] <= 0) or torch.any(on[:, -1:] <= 0):
            raise ValueError("Nonpositive affine weight")
        norm_off, norm_on = off[:, :-1] / off[:, -1:], on[:, :-1] / on[:, -1:]
        unit = torch.ones_like(off[:, -1:])
        matrix = torch.cat((norm_on[:, :9], norm_off[:, 9:], unit), dim=1)
        bias = torch.cat((norm_off[:, :9], norm_on[:, 9:], unit), dim=1)
        for role, coeffs in zip(STATES, (off, matrix, bias, on), strict=True):
            rgb = model.apply_coeffs(coeffs, content)
            if not torch.isfinite(rgb).all():
                raise ValueError("Nonfinite reconstructed render")
            images[role][start:stop] = byte_rgb(rgb)
    return {state: finish_spatial(target, frame, STRENGTH) for state, frame in images.items()}


def measured(images, target, authored, region, *, luma=False):
    desired = authored[region].astype(np.float64) - target[region]
    values = {key: image[region].astype(np.float64) for key, image in images.items()}
    if luma:
        desired = desired @ Y
        values = {key: value @ Y for key, value in values.items()}
    power = np.square(desired).sum()
    if power <= 0:
        raise ValueError("No measurable style signal")
    projections, errors = {}, {}
    for role in ("matrix", "bias", "on"):
        response = values[role] - values["off"]
        projections[role] = float((response * desired).sum() / power)
        errors[role] = float(np.abs(response - desired).mean())
    interaction = projections["on"] - projections["matrix"] - projections["bias"]
    return {"projection": projections, "response_mae": errors, "interaction_projection": interaction,
            "mean_marginal_projection": {role: projections[role] + .5 * interaction
                                         for role in ("matrix", "bias")}}


def compare(engine, out, prefix, target, references, controls, authored, regions, *, luma=False):
    images = render_states(engine, target, references)
    for state, (path, expected) in zip(("off", "on"), controls, strict=True):
        np.testing.assert_array_equal(images[state], checked_rgb(path, expected))
    return {
        "regions": {name: measured(images, target, authored, mask, luma=luma)
                    for name, mask in regions.items()},
        "output_sha256": {role: save(out / f"{prefix}-{role}.png", images[role]) for role in ("matrix", "bias")},
    }


def background(engine, out, originals):
    result = {}
    for name, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        folder = BASE / "bg-real-editor-look" / name
        assets = row["artifacts"]
        target, authored = (checked_rgb(folder / file, assets[file]) for file in ("target.png", "expected-on.png"))
        regions = {"whole": np.ones(target.shape[:2], dtype=bool)}
        for variant in ("C1", "F1"):
            tags = (f"{variant}-off", f"{variant}-on")
            references = [checked_rgb(folder / f"{tag}-reference.png", assets[f"{tag}-reference.png"])
                          for tag in tags]
            controls = [(BASE / FULL / "background" / name / f"{tag}.png",
                         originals[name]["outputs"][tag]["sha256"]) for tag in tags]
            result[f"{name}-{variant}"] = compare(engine, out, f"bg-{name}-{variant}", target, references,
                                                    controls, authored, regions, luma=True)
    return result


def film_b(engine, out, originals):
    prep = manifest(BASE, "arri-two-looks-v2")
    data = BASE / "arri-two-looks-v2"
    result = {}
    for key, row in manifest(BASE, "arri-film-b-crossed-v1")["cases"].items():
        scene, donor = row["target"], row["reference"]
        target, styled = endpoints(data, prep, scene)
        prefix = f"{scene}-from-{donor}"
        controls = [(BASE / FULL / "film_b" / f"{prefix}-{state}.png",
                     originals[key]["outputs"][state]["sha256"]) for state in ("off", "film_b")]
        regions = {"whole": np.ones(target.shape[:2], dtype=bool), "neutral": neutral(target) & neutral(styled)}
        result[key] = compare(engine, out, prefix, target, endpoints(data, prep, donor), controls,
                              finish_spatial(target, styled, STRENGTH), regions)
    return result


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    parent = verified_json(BASE / FULL / "results.json", FULL_SHA)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Expanded checkpoint changed")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    json_to(out / "results.json", {
        "parent_sha256": FULL_SHA, "checkpoint_sha256": WEIGHT_SHA256, "strength": STRENGTH,
        "background": background(engine, out, parent["suites"]["background"]["cases"]),
        "film_b": film_b(engine, out, parent["suites"]["film_b"]["cases"]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
