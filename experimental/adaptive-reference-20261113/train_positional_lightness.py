"""Fit equally sized positional and pooled heads on matched capture/edit pairs."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from probe_cross_source_wall_lightening_workbench import DIRECTIONS, OUTDOOR_SHA, frozen_scenes, load
from probe_matched_wall_lightening_workbench import PARENT
from probe_mid_matched_wall_cast import box
from probe_target_mirror_chroma import STRENGTH, digest, json_to
from reference_patch import enable_reference_patch, patch_grid
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


FAMILIES = {"mit": ("14n_copyroom1", "everett_living2"), "wildrelight": ("wall", "pillar")}
PROTOCOL = "d5e6a63ebde5e2c50a801bdb03970f8d9ca68de214b80f4a28976f0a24eb7ae0"
SEED = 20261126
STEPS = 128


class ConditionedEngine(SpatialEngine):
    def __init__(self, positioned):
        super().__init__("multiref-v1")
        torch.manual_seed(SEED)
        enable_reference_patch(self.model)
        self.head = self.model.reference_patch
        self.positioned = positioned
        axis = torch.linspace(-1, 1, 4, device=self.device) * (torch.pi / 2)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = .1 * torch.stack((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
        self.position = F.pad(coordinates.reshape(1, 16, 4), (124, 0))
        self.permute_tokens = False
        self.head.register_forward_pre_hook(self.add_position)
        if sum(item.numel() for item in self.head.parameters()) != 65664:
            raise ValueError("The two attention heads must have the frozen parameter count")

    def add_position(self, module, arguments):
        query, tokens = arguments
        if self.permute_tokens:
            tokens = tokens.flip((1,))
        if self.positioned:
            tokens = tokens + self.position
        return query, tokens

    def condition(self, target, reference):
        return patch_grid(self.model, self.api, target, reference, pooled=not self.positioned)

    @torch.inference_mode()
    def render(self, target, reference):
        if np.array_equal(target, reference):
            return target.copy()
        grid = self.condition(self.analysis(target), self.analysis(reference))
        return finish_spatial(target, PreparedSpatial(self, grid, {}).apply(target), STRENGTH)


def frames_for(scenes, family):
    photos, schedule = {}, []
    left, right = FAMILIES[family]
    for target, donor in ((left, right), (right, left)):
        for light, info in scenes[target]["frames"].items():
            if not info["metrics"]["admitted"] or not info.get("registered", True):
                continue
            for state in ("off", "on"):
                key = target, f"{light}-{state}"
                photos[key] = load(scenes[target], key[1])[1]
            for direction in DIRECTIONS[donor]:
                a, b = direction.split("-to-")
                schedule.append((target, light, donor, a, b))
    return photos, schedule


def inputs_for(engine, photos, scenes):
    patterns, pixels, edited = {}, {}, {}
    for key, image in photos.items():
        scene, tag = key
        patterns[key] = engine.analysis(image)[0]
        pixels[key] = F.interpolate(engine.tensor(image), (128, 128), mode="bilinear", align_corners=False,
                                    antialias=True)[0]
        if tag.endswith("-off") and scene not in edited:
            native = box(image, scenes[scene]["boxes"]["wall_left"])
            mask = torch.from_numpy(native).to(engine.device, dtype=torch.float32)[None, None]
            edited[scene] = F.interpolate(mask, (128, 128), mode="nearest")[0, 0].bool()
            if not edited[scene].any():
                raise ValueError("The author edit rectangle vanished during downsampling")
    return patterns, pixels, edited


def regional_loss(predicted, goal, edit):
    difference = (predicted - goal).abs()
    response = ((predicted[2:] - predicted[:2]) - (goal[2:] - goal[:2])).abs()
    lighting = (predicted[[1, 3]] - predicted[[0, 2]]).abs()
    return difference.mean() + difference[..., edit].mean() + .5 * response[..., edit].mean() + .5 * lighting.mean()


def flip(image, horizontal):
    return image.flip((-1,)) if horizontal else image


def run_arm(scenes, family, label, directory, steps, deadline):
    engine = ConditionedEngine(label == "local")
    photos, schedule = frames_for(scenes, family)
    patterns, pixels, edited = inputs_for(engine, photos, scenes)
    del photos
    rng = np.random.default_rng(SEED)
    orders = []
    while len(orders) < steps:
        orders.extend(rng.permutation(len(schedule)).tolist())
    optimizer = torch.optim.AdamW(engine.head.parameters(), lr=.0003, weight_decay=.0001)
    a0, a1 = schedule[0][:2], schedule[0][2:]
    target = patterns[(a0[0], f"{a0[1]}-off")][None]
    reference = patterns[(a1[0], f"{a1[1]}-on")][None]
    with torch.inference_mode():
        initial = engine.condition(target, reference)
        original = patch_grid(engine.model, engine.api, target, reference, disabled=True)
        if not torch.equal(original, initial):
            raise ValueError("Initial head changed the frozen grid")
    torch.cuda.reset_peak_memory_stats(engine.device)
    started = perf_counter()
    history, draw = [], []
    for index in orders[:steps]:
        if monotonic() > deadline:
            raise TimeoutError("Exceeded the fixed study runtime budget")
        target, light, donor, a, b = schedule[index]
        target_flip, donor_flip = rng.integers(0, 2, size=2).tolist()
        orig, styled = ((target, f"{light}-{state}") for state in ("off", "on"))
        refs = [(donor, f"{level}-{state}") for state in ("off", "on") for level in (a, b)]
        ref = flip(torch.stack([patterns[key] for key in refs]), donor_flip)
        full = flip(patterns[orig][None].expand(4, -1, -1, -1), target_flip)
        small = flip(pixels[orig][None].expand(4, -1, -1, -1), target_flip)
        truth = flip(torch.stack([pixels[orig], pixels[orig], pixels[styled], pixels[styled]]), target_flip)
        truth = torch.lerp(small, truth, STRENGTH)
        edit = flip(edited[target], target_flip)
        grid = engine.condition(full, ref)
        native = engine.model.apply_coeffs(engine.model.sampler(grid, small), small).clamp(0, 1)
        loss = regional_loss(torch.lerp(small, native, STRENGTH), truth, edit)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
        draw.append((schedule[index], target_flip, donor_flip))
    with torch.inference_mode():
        normal = engine.condition(full[:1], ref[:1])
        engine.permute_tokens = True
        permuted = engine.condition(full[:1], ref[:1])
        engine.permute_tokens = False
        maximum = float((normal - permuted).abs().max())
    if (label == "local" and maximum <= 0) or (label == "pooled" and maximum != 0):
        raise ValueError(f"Reference position probe contradicts {label} semantics: {maximum}")
    elapsed = perf_counter() - started
    weights = directory / f"{label}.pth"
    torch.save(engine.head.state_dict(), weights)
    record = dict(steps=steps, combinations=len(schedule), sample_order=sha256(repr(draw).encode()).hexdigest(),
                  seconds=elapsed, peak_cuda_bytes=torch.cuda.max_memory_allocated(engine.device),
                  first_loss=history[0], last_loss=history[-1], token_order_maximum=maximum,
                  head_sha256=digest(weights), parent=engine.checkpoint_sha)
    print(f"{family} {label}: {record}", flush=True)
    return record


def run(base, out, preflight):
    torch.set_num_threads(4)
    code = Path(__file__)
    if digest(code.with_name("positional-lightness-trial-protocol.md")) != PROTOCOL or STRENGTH != .8:
        raise ValueError("Changed position study protocol or strength")
    out.mkdir()
    selections = ("mit",) if preflight else tuple(FAMILIES)
    scenes = frozen_scenes(base)
    deadline = monotonic() + 4 * 3600
    result = {}
    for family in selections:
        directory = out / family
        directory.mkdir()
        row = {}
        for label in ("pooled", "local"):
            row[label] = run_arm(scenes, family, label, directory, 2 if preflight else STEPS, deadline)
        if row["pooled"]["sample_order"] != row["local"]["sample_order"]:
            raise ValueError("Training arms received different draws")
        result[family] = row
    json_to(out/"training.json", dict(preflight=preflight, indoor=PARENT, outdoor=OUTDOOR_SHA,
             strength=STRENGTH, protocol=PROTOCOL, script=digest(code), folds=result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--preflight", action="store_true")
    run(**vars(parser.parse_args()))
