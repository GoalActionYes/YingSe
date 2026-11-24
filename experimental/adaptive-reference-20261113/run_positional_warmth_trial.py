"""Preflight equally sized warm-color heads on the frozen two-family split."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from probe_mid_matched_wall_cast import box
from probe_playraw_flash_room import read
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, digest, json_to
from reference_patch import patch_grid
from train_positional_lightness import ConditionedEngine, SEED, flip, regional_loss


MANIFEST_SHA = "bbdb9b4bce5f1f06afc2c06deed59f29c21e1ccf1da64a1c9c5a17159d3b794e"
MANIFEST = Path(__file__).resolve().parents[2] / "docs/research/2026-11-24-warmth-source-holdout-evidence.json"
ROLES = ("base", "captured", "edited", "captured_edited")


def read_manifest():
    raw = MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != MANIFEST_SHA or STRENGTH != 0.8:
        raise ValueError("Changed holdout input manifest or strength")
    return json.loads(raw)


def read_photo(base, entry, family):
    file = base / entry["path"]
    if entry["family"] != family or digest(file) != entry["png_sha256"]:
        raise ValueError(f"Changed input file or family: {file}")
    rgb, _, image = read(file, False)
    if any(image[name] != entry[name] for name in ("icc_sha256", "pixels_sha256")):
        raise ValueError(f"Changed input pixel identity: {file}")
    return rgb


def load_case(base, manifest, family):
    spec = manifest["families"][family]
    reference_spec = spec["reference"]
    target_spec = spec["target"]
    references = {role: read_photo(base, reference_spec[role], family) for role in ROLES}
    targets = {state: read_photo(base, target_spec["photos"][state], family) for state in ("off", "on")}
    if reference_spec["base"]["scene"] == target_spec["photos"]["off"]["scene"]:
        raise ValueError("Target and reference must be different scenes")
    return targets, references, target_spec


def training_tensors(engine, targets, references, spec):
    original = targets["off"]
    content = engine.analysis(original).expand(4, -1, -1, -1)
    ref = torch.cat([engine.analysis(references[role]) for role in ROLES])
    off, on = [F.interpolate(engine.tensor(targets[state]), (128, 128), mode="bilinear",
                             align_corners=False, antialias=True) for state in ("off", "on")]
    source = off.expand(4, -1, -1, -1)
    goal = torch.lerp(source, torch.cat((off, off, on, on)), STRENGTH)
    mask = box(original, spec["boxes"]["wall_left"])
    mask = torch.from_numpy(mask).to(engine.device, dtype=torch.float32)[None, None]
    edit = F.interpolate(mask, (128, 128), mode="nearest")[0, 0].bool()
    if not edit.any() or torch.equal(source[..., edit], goal[..., edit]):
        raise ValueError("The authored target signal vanished during downsampling")
    return content, ref, source, goal, edit


def fit_arm(case, label, steps, deadline):
    began = monotonic()
    engine = ConditionedEngine(label == "local")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13-series parent checkpoint")
    trainable = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
    if [id(item) for item in trainable] != [id(item) for item in engine.head.parameters()]:
        raise ValueError("Unexpected trainable parameters outside the reference head")
    content, reference, source, goal, edit = training_tensors(engine, *case)
    with torch.inference_mode():
        zero = engine.condition(content, reference)
        frozen = patch_grid(engine.model, engine.api, content, reference, disabled=True)
        if not torch.equal(zero, frozen):
            raise ValueError("Initial reference head changed the frozen grid")

    optimizer = torch.optim.AdamW(engine.head.parameters(), lr=0.0003, weight_decay=0.0001)
    draw = np.random.default_rng(SEED).integers(0, 2, size=(steps, 2))
    torch.cuda.synchronize(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    started = monotonic()
    losses = []
    for mirrored, ref_mirrored in draw.tolist():
        if monotonic() > deadline:
            raise TimeoutError("Exceeded the training and evaluation time ceiling")
        target = flip(content, mirrored)
        ref = flip(reference, ref_mirrored)
        small = flip(source, mirrored)
        grid = engine.condition(target, ref)
        native = engine.model.apply_coeffs(engine.model.sampler(grid, small), small).clamp(0, 1)
        loss = regional_loss(torch.lerp(small, native, STRENGTH), flip(goal, mirrored), flip(edit, mirrored))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.head.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(engine.device)
    fit_seconds = monotonic() - started
    if not np.isfinite(losses).all() or not all(torch.isfinite(item).all() for item in engine.head.parameters()):
        raise ValueError("Nonfinite loss or fitted head")
    with torch.inference_mode():
        normal = engine.condition(content, reference)
        engine.permute_tokens = True
        permuted = engine.condition(content, reference)
        engine.permute_tokens = False
        difference = float((normal - permuted).abs().max())
    if (label == "local" and difference <= 0) or (label == "pooled" and difference != 0):
        raise ValueError(f"Unexpected position signal for the {label} arm: {difference}")
    row = dict(steps=steps, sample_order_sha256=sha256(draw.tobytes()).hexdigest(), initial_grid_exact=True,
               first_loss=losses[0], final_loss=losses[-1], order_effect=difference,
               optimizer_seconds=fit_seconds, wall_seconds=monotonic() - began,
               peak_cuda_bytes=torch.cuda.max_memory_allocated(engine.device), checkpoint=engine.checkpoint_sha)
    return engine, row


def run(base, out):
    start = monotonic()
    torch.set_num_threads(4)
    manifest = read_manifest()
    out.mkdir()
    rows = {}
    for fold in manifest["folds"]:
        train, holdout = fold["train"], fold["held_out"]
        if train == holdout:
            raise ValueError("The training and held-out families must differ")
        case = load_case(base, manifest, train)
        arms = {}
        for label in ("pooled", "local"):
            engine, row = fit_arm(case, label, 2, start + 20 * 60)
            arms[label] = row
            print(train, label, row, flush=True)
            del engine
        if arms["pooled"]["sample_order_sha256"] != arms["local"]["sample_order_sha256"]:
            raise ValueError("The two arms saw different training draws")
        rows[train] = dict(held_out=holdout, arms=arms)
    script = Path(__file__)
    json_to(out / "preflight.json", dict(training_manifest=MANIFEST_SHA, preflight=True, script_sha256=digest(script),
                                        protocol_sha256=digest(script.with_name("positional-warmth-trial-protocol.md")),
                                        total_wall_seconds=monotonic() - start, folds=rows))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
