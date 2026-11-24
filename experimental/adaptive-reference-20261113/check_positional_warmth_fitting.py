"""Check the native-size fit on the same families used to train the frozen heads."""

from argparse import ArgumentParser
from functools import partial
from itertools import product
from pathlib import Path
from time import monotonic

import torch

from evaluate_positional_lightness import baseline
from evaluate_positional_warmth_trial import ARMS, PROTOCOL, STEPS, heldout_scores, render_case
from probe_matched_wall_warmth_workbench import target_areas
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to
from run_positional_warmth_trial import MANIFEST_SHA, load_case, read_manifest, training_tensors
from spatial_render import SpatialEngine, finish_spatial
from train_positional_lightness import ConditionedEngine, flip


FORMAL_SHA = "4645a0a72445224930e2e4d9a9e768f4c3d65d49f1e5d4a2c609c6c7349b8d8d"
FITS_SHA = "b8dbc93f20dc5690fab4ff2cf8164ee2c26e34e9ea762e958138b07ae1d4cd20"


@torch.inference_mode()
def scores_at_training_size(engine, case):
    content, references, source, goal, edit = training_tensors(engine, *case)
    scores = {}
    for target_flip, ref_flip in product((False, True), repeat=2):
        grid = engine.condition(flip(content, target_flip), flip(references, ref_flip))
        small = flip(source, target_flip)
        raw = engine.model.apply_coeffs(engine.model.sampler(grid, small), small).clamp(0, 1)
        output = torch.lerp(small, raw, STRENGTH)
        area = flip(edit, target_flip)
        expected = flip(goal, target_flip)
        deltas = (expected[2:] - expected[:2])[..., area].double()
        observed = (output[2:] - output[:2])[..., area].double()
        energy = deltas.square().sum(dim=(1, 2))
        if not bool((energy > 0).all()):
            raise ValueError("Empty authored signal in the training-size edit region")
        projections = (observed * deltas).sum(dim=(1, 2)) / energy
        error = (observed - deltas).abs().mean(dim=(1, 2)) * 255
        scores[f"target-{int(target_flip)}/reference-{int(ref_flip)}"] = dict(
            projection=projections.tolist(), response_mae_rgb8=error.tolist(),
            passed=((projections >= .75) & (projections <= 1.25) & (error <= 3)).tolist())
    return scores


def run(base, formal, out, training_size_only):
    start = monotonic()
    deadline = start + 120
    torch.set_num_threads(4)
    manifest = read_manifest()
    previous = checked_json(formal / "results.json", FORMAL_SHA)
    fits = checked_json(formal / "training/fits.json", FITS_SHA)
    if fits["manifest"] != MANIFEST_SHA or fits["protocol"] != PROTOCOL or previous["training"] != fits["folds"]:
        raise ValueError("Changed training provenance")

    engine13 = None if training_size_only else SpatialEngine("multiref-v1")
    if engine13 is not None and engine13.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed baseline checkpoint")
    out.mkdir()
    records = {}
    for fold in manifest["folds"]:
        family = fold["train"]
        case = load_case(base, manifest, family)
        targets, refs, spec = case
        original = targets["off"]
        expected = finish_spatial(original, targets["on"], STRENGTH)
        areas = target_areas(original, spec, False)
        arms = {}
        for label in ARMS[1:] if training_size_only else ARMS:
            if label == "13":
                render = partial(baseline, engine=engine13)
            else:
                details = fits["folds"][family][label]
                checkpoint = formal / "training" / f"{family}-{label}.pth"
                if digest(checkpoint) != details["sha256"] or details["steps"] != STEPS:
                    raise ValueError(f"Changed fitted head {family}/{label}")
                engine = ConditionedEngine(label == "local")
                engine.head.load_state_dict(torch.load(checkpoint, map_location=engine.device, weights_only=True),
                                            strict=True)
                if training_size_only:
                    arms[label] = scores_at_training_size(engine, case)
                    continue
                render = engine.render
            images, receipt = render_case(render, original, expected, refs, out / family / label, deadline)
            if not receipt["identity"]["exact"]:
                raise ValueError(f"Same-image control failed for {family}/{label}")
            measured = heldout_scores(original, expected, images, areas)
            passed = all(area["passed"] for area in measured.values())
            arms[label] = dict(complete=passed, measured=measured, **receipt)
        records[family] = arms
    scope = "training-size same-family edit response" if training_size_only else "same-family fit diagnosis"
    json_to(out / "results.json", dict(scope=scope, formal_sha256=FORMAL_SHA,
                                       fits_sha256=FITS_SHA, checkpoint=CHECKPOINT, families=records))
    print(f"Recorded {scope} in {monotonic() - start:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "formal", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--training-size-only", action="store_true")
    run(**vars(parser.parse_args()))
