"""Measure frozen spatial-grid capacity on known local edits with pixel-block holdout."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
import torch
import torch.nn.functional as F

from capacity_audit import block_split
from probe_arri_shared_look import metrics
from probe_gros_grid_expression import rendering_features
from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, areas_and_weight
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


PARENT = "0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7"
GRID = 16
LAYERS = 8
FEATURES = 4
TERMS = GRID * GRID * LAYERS * FEATURES


def axis_weights(count, device):
    positions = (2 * torch.arange(count, dtype=torch.float32, device=device) + 1) / count - 1
    coordinates = torch.stack((positions, torch.zeros_like(positions)), dim=-1)[None, None]
    basis = torch.eye(GRID, dtype=torch.float32, device=device).reshape(1, GRID, 1, GRID)
    weights = F.grid_sample(basis, coordinates, align_corners=False, padding_mode="border")
    return weights[0, :, 0].T.cpu().numpy()


def support(weights):
    chosen = np.argsort(weights, axis=-1)[:, -2:]
    values = np.take_along_axis(weights, chosen, axis=-1)
    np.testing.assert_allclose(values.sum(axis=-1), 1, atol=1e-6, rtol=0)
    return chosen, values


def spatial_features(affine, depth, x_axis, y_axis, rows):
    width = affine.shape[1]
    iy, ix = np.divmod(rows, width)
    z, zw = support(depth[iy, ix])
    x, xw = x_axis
    y, yw = y_axis
    corner = ((z[:, :, None, None] * GRID + y[iy, None, :, None]) * GRID + x[ix, None, None, :])
    weight = zw[:, :, None, None] * yw[iy, None, :, None] * xw[ix, None, None, :]
    terms = corner[..., None] * FEATURES + np.arange(FEATURES)
    basis = weight[..., None] * affine[iy, ix, None, None, None, :]
    count = len(rows)
    indices = terms.reshape(-1)
    indptr = np.arange(count + 1) * (8 * FEATURES)
    matrix = csr_matrix((basis.reshape(-1), indices, indptr), shape=(count, TERMS))
    matrix.eliminate_zeros()
    return matrix


def prepare(engine, coefficients):
    nodes = coefficients.reshape(LAYERS, GRID, GRID, FEATURES, 3)
    matrices = nodes[..., :3, :].swapaxes(-2, -1).reshape(LAYERS, GRID, GRID, 9)
    field = np.concatenate((matrices, nodes[..., 3, :], np.ones((LAYERS, GRID, GRID, 1))), axis=-1)
    native = torch.as_tensor(np.moveaxis(field, -1, 0).copy(), device=engine.device, dtype=torch.float32)[None]
    return PreparedSpatial(engine, native, {})


@torch.inference_mode()
def compare_renderer(engine, original, native, design, coefficients, rows):
    prediction = torch.as_tensor(design @ coefficients, device=engine.device, dtype=torch.float32)
    pixels = engine.model.post_guide(prediction.T.reshape(1, 3, 1, -1).clamp(-1, 1))
    direct = native.float_output(original).reshape(-1, 3)
    expected = pixels[0, :, 0].T.cpu().numpy()
    difference = float(np.abs(expected - direct[rows]).max())
    if difference > 1e-5:
        raise ValueError(f"Sparse design differs from the native renderer: {difference}")
    return difference


def fit(affine, depth, inverse, rows, design):
    width = affine.shape[1]
    y, x = np.divmod(rows, width)
    shared = (depth[y, x, :, None] * affine[y, x, None, :]).reshape(len(rows), -1)
    truth = inverse[y, x]
    coefficients, _, rank, _ = np.linalg.lstsq(shared, truth, rcond=1e-5)
    if rank < 4:
        raise ValueError("Insufficient rank for the shared color model")
    start = np.broadcast_to(coefficients.reshape(LAYERS, 1, 1, FEATURES, 3),
                            (LAYERS, GRID, GRID, FEATURES, 3)).reshape(TERMS, 3).copy()
    residual = truth - design @ start
    norm = np.sqrt(np.asarray(design.power(2).sum(axis=0)).ravel())
    scale = 1 / np.maximum(.1, norm)
    normalized = design.multiply(scale).tocsr()
    solved = [lsmr(normalized, residual[:, channel], damp=.01, atol=1e-6, btol=1e-6, maxiter=240)
              for channel in range(3)]
    refined = start + scale[:, None] * np.stack([entry[0] for entry in solved], axis=1)
    if not np.isfinite(refined).all():
        raise ValueError("Nonfinite fitted grid")
    audit = dict(shared_rank=int(rank), active_terms=int((norm > 0).sum()), train_pixels=len(rows),
                 stop=[int(entry[1]) for entry in solved], iterations=[int(entry[2]) for entry in solved],
                 residual=[float(entry[3]) for entry in solved], max_coefficient=float(np.abs(refined).max()))
    return {"shared": start, "spatial": refined}, audit


def evaluate(original, expected, grade, regions):
    results = {}
    for name, mask in regions.items():
        row = metrics(original, expected, original, grade, mask)
        if name.startswith("screen") or name == "outside":
            row["passed"] = row["on_matches_authored_mae"] <= 3
        diff = np.abs(grade[mask].astype(np.int16) - expected[mask])
        results[name] = dict(**row, p95=float(np.percentile(diff, 95)), pixels=int(mask.sum()))
    return results


def render_case(engine, original, edit, regions, control):
    h, w = original.shape[:2]
    affine, depth, inverse = rendering_features(engine, original, edit)
    x_axis, y_axis = support(axis_weights(w, engine.device)), support(axis_weights(h, engine.device))
    even = block_split(h, w, 16).reshape(h, w)
    y, x = np.ogrid[:h, :w]
    sampled = (y % 3 == 0) & (x % 3 == 0)
    verification = np.flatnonzero((y % 9 == 1) & (x % 7 == 1))
    bridge = spatial_features(affine, depth, x_axis, y_axis, verification)
    checks = {}
    if control:
        rng = np.random.default_rng(4)
        test = rng.normal(0, .08, (TERMS, 3))
        base = prepare(engine, test)
        checks["random_grid_max_error"] = compare_renderer(engine, original, base, bridge, test, verification)
        np.testing.assert_array_equal(base.apply(original, strength=0), original)
    both = {arm: original.copy() for arm in ("shared", "spatial")}
    audits = []
    for train in (even, ~even):
        rows = np.flatnonzero(train & sampled)
        design = spatial_features(affine, depth, x_axis, y_axis, rows)
        fitted, audit = fit(affine, depth, inverse, rows, design)
        audit["bridge_max_error"] = {}
        for arm, coefficients in fitted.items():
            native = prepare(engine, coefficients)
            audit["bridge_max_error"][arm] = compare_renderer(engine, original, native, bridge, coefficients, verification)
            grade = finish_spatial(original, native.apply(original), STRENGTH)
            both[arm][~train] = grade[~train]
        audits.append(audit)
    expected = finish_spatial(original, edit, STRENGTH)
    return both, expected, audits, checks


def run(project, output):
    if scipy.__version__ != "1.18.1":
        raise ValueError("Use the frozen SciPy wheel from the protocol")
    source = project / "gros-local-editor-channel-v2"
    parent = checked_json(source / "results.json", PARENT)
    script = Path(__file__)
    coords = frozen_json(script.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or parent["strength"] != STRENGTH:
        raise ValueError("Changed frozen native renderer")
    output.mkdir()
    cases = {}
    for subject, lights in parent["intake"].items():
        areas, weight = areas_and_weight(coords[subject], coords["screen"])
        areas["outside"] = weight == 0
        for light, record in lights.items():
            slug = f"{subject}-{light}"
            original, edit = [read_frame(source / subject / light / f"{state}.png", record["hashes"][state])
                              for state in ("off", "on")]
            rendered, expected, folds, controls = render_case(engine, original, edit, areas, not cases)
            scores = {arm: evaluate(original, expected, frame, areas) for arm, frame in rendered.items()}
            images = {"original": original, "expected": expected} | rendered
            cases[slug] = dict(scores=scores, all_pass={arm: all(row["passed"] for row in rows.values())
                                                      for arm, rows in scores.items()},
                               folds=folds, controls=controls,
                               hashes={arm: save(output / f"{slug}-{arm}.png", frame) for arm, frame in images.items()},
                               panel=panel(output / f"{slug}-comparison.png", {"same image": images}))
            print(slug, cases[slug]["all_pass"], flush=True)
    json_to(output / "results.json", dict(parent=PARENT, checkpoint=CHECKPOINT, strength=STRENGTH, scipy=scipy.__version__,
            script=digest(script), protocol=digest(script.with_name("gros-local-grid-capacity-protocol.md")), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
