"""Test a frozen decoder with constant and position-weighted context offsets."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from capacity_audit import block_split
from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, areas_and_weight
from probe_gros_local_grid_capacity import evaluate
from probe_gros_mixed_light_exposure_toggle import CHEEKS, LEVELS, LOCAL_EDIT, TRANSFER
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


PARENT = "a12aaa38a01e0ca459b7cfbc0e2c555420c020159b7399e66fc81f9b551bf6b0"
STEPS = 160
ARMS = ("shared", "position")


def histogram(engine, image):
    return engine.api["batch_histogram_1D"](engine.api["srgb_to_ycbcr"](image))


def decode(engine, features, context):
    grid = engine.model.fusion(torch.cat((features, context), dim=1)).reshape(1, 13, 8, 16, 16)
    alpha = F.softplus(grid[:, -1:]) + .001
    return engine.api["lowpass"](torch.cat((grid[:, :-1] * alpha, alpha), dim=1))


def site_plane(array, device):
    plane = torch.from_numpy(array.astype(np.float32))[None, None].to(device)
    return F.adaptive_avg_pool2d(plane, (16, 16))


@torch.no_grad()
def baseline_contexts(engine, original, donor, case, light_root, edit_root, lights, edits):
    content = engine.analysis(original)
    low = engine.api["TF"].resize(content, (64, 64), antialias=True)
    features = engine.model.img_projection(low)
    content_hist = histogram(engine, content)
    contexts, evidence = {}, {}
    for light in LEVELS:
        for state in ("off", "on"):
            name = f"{case}-spatial13-{light}"
            if state == "on":
                wanted = read_frame(edit_root / f"{name}-on.png", edits["cases"][case]["spatial13"]["on"][light])
                reference = read_frame(edit_root / f"{donor}-{light}-editor-on.png", edits["intake"][donor][light]["output_sha256"])
            else:
                wanted = read_frame(light_root / f"{name}.png", lights["cases"][case]["spatial13"]["outputs"][light])
                reference = read_frame(light_root / f"reference-{donor}-{light}.png", lights["prepared_hashes"][donor][light])
            hist = torch.cat((content_hist, histogram(engine, engine.analysis(reference))), dim=1)
            token = engine.model.hist_projection(hist)
            if token.shape != (1, 128, 1):
                raise ValueError("Unexpected number of reference tokens")
            context = engine.model.attention(features, token)
            constant = context[:, :, :1, :1].expand_as(context)
            grid = decode(engine, features, constant)
            native = engine.prepare(original, reference)
            grid_diff = float((grid - native.grid).abs().max())
            context_diff = float((context - constant).abs().max())
            if grid_diff > 1e-6 or context_diff > 1e-6:
                raise ValueError("Split inference differs from original single-token path")
            grade = finish_spatial(original, PreparedSpatial(engine, grid, {}).apply(original), STRENGTH)
            np.testing.assert_array_equal(grade, wanted)
            contexts[light, state] = constant
            evidence[f"{light}-{state}"] = dict(grid_max=grid_diff, context_spatial_max=context_diff)
    changes = {light: float((contexts[light, "on"] - contexts[light, "off"]).square().mean().sqrt()) for light in LEVELS}
    capture = float((contexts["high", "off"] - contexts["low", "off"]).square().mean().sqrt())
    return features, contexts["medium", "off"], dict(native=evidence, edit_context_rms=changes, capture_context_rms=capture)


@torch.no_grad()
def sampling(engine, original, truth, selected, regions):
    h, w = original.shape[:2]
    y, x = np.nonzero(selected)
    content = engine.tensor(original[y, x][None])
    weights = F.softmax(engine.model.sampler.L_weights, dim=1)
    guide = (engine.model.sampler.pre_guide(content) * weights).sum(1).flatten()
    xx = torch.as_tensor(x, dtype=torch.float32, device=engine.device)
    yy = torch.as_tensor(y, dtype=torch.float32, device=engine.device)
    locations = torch.stack(((2 * xx + 1) / w - 1, (2 * yy + 1) / h - 1, (guide - .5) * 2), dim=-1)
    locations = locations[None, None, None].clamp(-1, 1)
    sign = (truth[y, x].astype(np.float32) - original[y, x]) / 255
    desired = torch.as_tensor(sign.T.copy(), device=engine.device)[None, :, None]
    masks = {name: torch.as_tensor(mask[y, x], device=engine.device) for name, mask in regions.items()}
    if any(mask.sum() < 12 for mask in masks.values()):
        raise ValueError("Insufficient samples in a frozen area")
    return content, locations, desired, masks


def sample(engine, grid, content, locations):
    coefficients = F.grid_sample(grid, locations, align_corners=False, padding_mode="border").squeeze(2)
    return engine.model.apply_coeffs(coefficients, content)


def fit(engine, original, features, base, plane, examples, check_native):
    content, locations, desired, masks = examples
    grid = decode(engine, features, base)
    anchored = sample(engine, grid, content, locations)
    if check_native is not None:
        full = PreparedSpatial(engine, grid, {}).float_output(original)
        native = full[check_native].T[None, :, None]
        np.testing.assert_allclose(anchored.detach().cpu().numpy(), native, atol=1e-5, rtol=0)
    offset = torch.zeros((1, 128, 1, 1), device=engine.device, requires_grad=True)
    optimizer = torch.optim.Adam([offset], lr=.025)
    weights = torch.stack([mask / mask.sum() for mask in masks.values()]).mean(dim=0)
    loss_log = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        grade = sample(engine, decode(engine, features, base + offset * plane), content, locations)
        errors = F.smooth_l1_loss((grade - anchored) * STRENGTH, desired, beta=2 / 255, reduction="none")
        loss = (errors.mean(dim=1).flatten() * weights).sum() + 1e-4 * offset.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            offset.div_(offset.square().mean().sqrt().clamp_min(1.))
        if step in (0, STEPS - 1):
            loss_log.append(float(loss.detach()))
    if not torch.isfinite(offset).all() or not np.isfinite(loss_log).all():
        raise ValueError("Nonfinite context fit")
    with torch.no_grad():
        fitted = decode(engine, features, base + offset * plane)
        return fitted, dict(loss_start_end=loss_log, rms=float(offset.square().mean().sqrt()), samples=int(desired.shape[-1]))


def run(base, out):
    torch.set_num_threads(4)
    light_root = base / "gros-mixed-light-transfer-repeat-v1"
    edit_root = base / "gros-mixed-light-exposure-toggle-v1"
    original_root = base / "gros-local-editor-channel-v2"
    lights = checked_json(light_root / "results.json", TRANSFER)
    edits = checked_json(edit_root / "results.json", PARENT)
    checked_json(original_root / "results.json", LOCAL_EDIT)
    if not edits["admitted"] or edits["checkpoint"] != CHECKPOINT or edits["strength"] != STRENGTH:
        raise ValueError("Changed development inputs")
    coords = frozen_json(Path(__file__).with_name("gros-local-editor-cross-regions.json"), REGIONS)
    out.mkdir()
    engine = SpatialEngine("multiref-v1")
    figures = {}
    for case in sorted(edits["cases"]):
        target, donor = case.split("-from-")
        original = read_frame(original_root / target / "cold" / "off.png", lights["prepared_hashes"][target]["target"])
        expected = read_frame(edit_root / f"{target}-target-editor-on.png", edits["intake"][target]["target"]["output_sha256"])
        regions, alpha = areas_and_weight(coords[target], coords["screen"])
        regions["outside"] = alpha == 0
        features, context, check = baseline_contexts(engine, original, donor, case, light_root, edit_root, lights, edits)
        plane = site_plane(alpha, engine.device)
        baseline = decode(engine, features, context)
        torch.testing.assert_close(decode(engine, features, context + torch.ones_like(context) * torch.zeros_like(plane)), baseline, atol=0, rtol=0)
        anchor = PreparedSpatial(engine, baseline, {}).apply(original).astype(np.int16)
        baseline_grade = finish_spatial(original, anchor.astype(np.uint8), STRENGTH)
        y, x = np.ogrid[:original.shape[0], :original.shape[1]]
        blocks = block_split(*original.shape[:2], 16).reshape(original.shape[:2])
        frames = {arm: original.copy() for arm in ARMS}
        receipts = []
        for fold, fitting in enumerate((blocks, ~blocks)):
            pixels = fitting & (y % 6 == 2) & (x % 6 == 2)
            examples = sampling(engine, original, expected, pixels, regions)
            row = {}
            for arm in ARMS:
                grid, row[arm] = fit(engine, original, features, context, plane if arm == "position" else 1., examples,
                                    pixels if arm == "shared" and fold == 0 else None)
                edited = PreparedSpatial(engine, grid, {}).apply(original).astype(np.int16)
                response = np.rint(original + (edited - anchor) * STRENGTH).clip(0, 255).astype(np.uint8)
                frames[arm][~fitting] = response[~fitting]
            receipts.append(row)
        results = {arm: evaluate(original, expected, rgb, regions) for arm, rgb in frames.items()}
        hashes = {arm: save(out / f"{case}-{arm}.png", rgb) for arm, rgb in frames.items()}
        board = panel(out / f"{case}-comparison.png", {"target": {"original": original, "expected": expected,
                      "native anchor": baseline_grade}, "anchored response": frames})
        figures[case] = dict(checks=check, fitting=receipts, scores=results, hashes=hashes, board=board,
                            complete={arm: all(row["passed"] for row in results[arm].values()) for arm in ARMS})
        print(case, figures[case]["complete"], flush=True)
    code = Path(__file__)
    json_to(out / "results.json", dict(parent=PARENT, transfer=TRANSFER, checkpoint=CHECKPOINT, steps=STEPS,
        cases=figures, script_sha256=digest(code), protocol_sha256=digest(code.with_name("gros-context-capacity-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
