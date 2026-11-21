"""Fit known edits into the frozen renderer, scoring on held pixel blocks."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from probe_arri_shared_look import metrics
from probe_gros_film_b_cross import GROUPS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)
from spatial_render import PreparedSpatial, coordinates


@torch.inference_mode()
def rendering_features(engine, target, edited):
    model = engine.model
    height, width = target.shape[:2]
    rgb = engine.tensor(target)
    guide = (model.sampler.pre_guide(rgb) * F.softmax(model.sampler.L_weights, dim=1)).sum(1)
    locations = coordinates(width, height, 0, height, guide)
    depth_basis = torch.eye(8, device=engine.device).reshape(1, 8, 8, 1, 1)
    interpolated = F.grid_sample(depth_basis, locations, align_corners=False, padding_mode="border")
    weights = interpolated[0, :, 0].permute(1, 2, 0).cpu().numpy()
    if not np.allclose(weights.sum(-1), 1, atol=1e-6, rtol=0):
        raise ValueError("Unexpected depth interpolation")
    features = model.pre_guide(rgb * 2 - 1)[0].permute(1, 2, 0).cpu().numpy()
    affine = np.concatenate((features, np.ones((*features.shape[:2], 1))), axis=-1)
    nodes = torch.linspace(-1, 1, 64, device=engine.device)
    display = model.post_guide(nodes.expand(1, 3, 1, 64))[0, :, 0].cpu().numpy()
    if not (np.diff(display, axis=1) > 0).all():
        raise ValueError("Frozen display mapping is not strictly monotonic")
    inverse = np.stack([
        np.interp(edited[..., channel] / 255, display[channel], nodes.cpu().numpy())
        for channel in range(3)
    ], axis=-1)
    return affine, weights, inverse


def solve_grid(affine, weights, inverse, train):
    design = (weights[..., None] * affine[..., None, :]).reshape(*train.shape, -1)
    coefficients, _, rank, _ = np.linalg.lstsq(design[train], inverse[train], rcond=1e-5)
    if rank < 4 or not np.isfinite(coefficients).all():
        raise ValueError("Degenerate fit")
    return coefficients, rank


def render_grid(engine, target, coefficients):
    matrices = coefficients.reshape(-1, 4, 3)
    weights_rgb = matrices[:, :3, :].transpose(0, 2, 1).reshape(-1, 9)
    nodes = np.column_stack((weights_rgb, matrices[:, 3, :], np.ones(len(matrices))))
    tensor = torch.as_tensor(nodes.T.copy(), dtype=torch.float32, device=engine.device)
    grid = tensor.reshape(1, 13, -1, 1, 1).expand(1, 13, 8, 16, 16).contiguous()
    return PreparedSpatial(engine, grid, {})


def fit_and_render(engine, target, affine, weights, inverse, train):
    coefficients, rank = solve_grid(affine, weights, inverse, train)
    prepared = render_grid(engine, target, coefficients)
    design = (weights[..., None] * affine[..., None, :]).reshape(*train.shape, -1)
    prediction = design @ coefficients
    with torch.inference_mode():
        values = torch.as_tensor(prediction, device=engine.device, dtype=torch.float32)
        display = engine.model.post_guide(values.clamp(-1, 1).permute(2, 0, 1)[None])
    display = display[0].permute(1, 2, 0).cpu().numpy()
    disagreement = float(np.max(np.abs(display - prepared.float_output(target))))
    if disagreement > 1e-5:
        raise ValueError(f"Fitted design disagrees with the original renderer by {disagreement}")
    return finish_spatial(target, prepared.apply(target), STRENGTH), rank, disagreement


def evaluate(target, expected, result, regions, subset):
    # The same known response is evaluated once. Lamp differences are not measured here.
    scores = {}
    for name, mask in regions.items():
        selected = mask & subset
        count = int(selected.sum())
        if count < 64:
            raise ValueError(f"Insufficient pixels in {name}: {count}")
        scores[name] = metrics(target, expected, target, result, selected) | {"pixels": count}
    return scores


def run(out):
    source = BASE / "gros-film-b-cross-v2"
    archive = checked_json(source / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or archive["strength"] != STRENGTH:
        raise ValueError("Frozen renderer or strength changed")
    out.mkdir()
    result = {}
    for person in GROUPS:
        key = f"{person}-cold"
        source_hashes = archive["inputs"][key]["hashes"]
        original, edited = (
            checked_rgb(source / f"{key}-{state}.png", source_hashes[state]) for state in ("off", "on")
        )
        expected = finish_spatial(original, edited, STRENGTH)
        zones = masks(labels[person]["cold"], labels["screen"], original.shape[:2])
        y, x = np.ogrid[:original.shape[0], :original.shape[1]]
        train = (y // 16 + x // 16) % 2 == 0
        features, depth_weights, inverse = rendering_features(engine, original, edited)
        images, report = {}, {}
        for arm, weights in (("shared", np.ones_like(depth_weights[..., :1])), ("depth", depth_weights)):
            image, rank, agreement = fit_and_render(engine, original, features, weights, inverse, train)
            images[arm] = image
            report[arm] = dict(rank=int(rank), design_render_max_difference=agreement,
                               train=evaluate(original, expected, image, zones, train),
                               held=evaluate(original, expected, image, zones, ~train),
                               sha256=save(out / f"{person}-{arm}.png", image))
        board = panel(out / f"{person}-comparison.png", {
            "expected": {"original": original, "known Film B": expected}, "fit": images,
        })
        prior = {}
        for direction, record in archive["cases"].items():
            if direction.startswith(f"{person}-from-"):
                predictions = {
                    light: {state: checked_rgb(source / f"{direction}-13-{light}-{state}.png",
                                               record["hashes"][f"13-{light}-{state}"])
                            for state in ("off", "on")}
                    for light in ("cold", "warm")
                }
                prior[direction] = score(original, expected, predictions,
                                         {name: mask & ~train for name, mask in zones.items()})
        result[person] = dict(fitted=report, baseline_13_on_held=prior, board=board)
    protocol = Path(__file__).with_name("gros-grid-expression-protocol.md")
    json_to(out / "results.json", dict(parent=BASELINE_SHA, regions=REGIONS_SHA, checkpoint=CHECKPOINT,
                                       strength=STRENGTH, protocol=digest(protocol), cases=result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
