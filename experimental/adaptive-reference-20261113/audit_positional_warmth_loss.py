"""Measure the frozen warm-color objective in output and reference-head space."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic

import torch

from check_positional_warmth_fitting import FITS_SHA, FORMAL_SHA
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to
from run_positional_warmth_trial import load_case, read_manifest, training_tensors
from train_positional_lightness import ConditionedEngine, regional_loss


TERMS = ("whole", "region", "edit_response", "lighting")
WEIGHTS = (1.0, 1.0, 0.5, 0.5)
PARAMETER_RMS = 0.0001


def residuals(output, goal, edit):
    difference = output - goal
    response = (output[2:] - output[:2]) - (goal[2:] - goal[:2])
    lighting = output[[1, 3]] - output[[0, 2]]
    return difference, difference[..., edit], response[..., edit], lighting


def terms(output, goal, edit):
    return {name: weight * value.abs().mean()
            for name, weight, value in zip(TERMS, WEIGHTS, residuals(output, goal, edit))}


def values(scores):
    numbers = {name: float(value.detach()) for name, value in scores.items()}
    return {**numbers, "total": sum(numbers.values())}


def projection(output, goal, edit):
    authored = (goal[2:] - goal[:2])[..., edit].double()
    observed = (output[2:] - output[:2])[..., edit].double()
    return ((observed * authored).sum((1, 2)) / authored.square().sum((1, 2))).tolist()


def forward(engine, content, references, source):
    grid = engine.condition(content, references)
    adjusted = engine.model.apply_coeffs(engine.model.sampler(grid, source), source).clamp(0, 1)
    return torch.lerp(source, adjusted, STRENGTH)


@torch.inference_mode()
def output_probe(output, goal, edit):
    output, goal = output.double(), goal.double()
    wanted = output[:2, ..., edit] + (goal[2:] - goal[:2])[..., edit]
    corrected = output.clone()
    corrected[2:, ..., edit] = wanted.clamp(0, 1)
    before, after = residuals(output, goal, edit), residuals(corrected, goal, edit)
    directional = {}
    for name, weight, old, new in zip(TERMS, WEIGHTS, before, after):
        delta = new - old
        slope = torch.where(old == 0, delta.abs(), old.sign() * delta)
        directional[name] = weight * slope.mean()
    return dict(start=values(terms(output, goal, edit)), corrected=values(terms(corrected, goal, edit)),
                projection_start=projection(output, goal, edit), projection_corrected=projection(corrected, goal, edit),
                clipped_channels=int(((wanted < 0) | (wanted > 1)).sum()), edited_channels=wanted.numel(),
                right_derivative=values(directional))


def parameter_probe(engine, content, references, source, goal, edit):
    output = forward(engine, content, references, source)
    computed = terms(output, goal, edit)
    torch.testing.assert_close(sum(computed.values()), regional_loss(output, goal, edit), atol=1e-7, rtol=1e-6)
    parameters = tuple(engine.head.parameters())
    gradients = {}
    for name, term in computed.items():
        rows = torch.autograd.grad(term, parameters, retain_graph=name != TERMS[-1])
        gradients[name] = torch.cat([row.detach().flatten() for row in rows]).double()
    norms = {name: float(vector.norm()) for name, vector in gradients.items()}
    response = gradients["edit_response"]
    rms = response.square().mean().sqrt()
    if not bool(rms > 0):
        return dict(losses=values(computed), norms=norms, edit_gradient_zero=True), output.detach()

    direction = -response / rms
    derivative = values({name: gradient @ direction for name, gradient in gradients.items()})
    origin = [parameter.detach().clone() for parameter in parameters]
    pieces = direction.split([parameter.numel() for parameter in parameters])
    displaced = {}
    with torch.no_grad():
        try:
            for sign in (1, -1):
                for parameter, before, delta in zip(parameters, origin, pieces):
                    change = delta.reshape_as(parameter).to(dtype=parameter.dtype)
                    parameter.copy_(before + (sign * PARAMETER_RMS) * change)
                moved = forward(engine, content, references, source)
                displaced[sign] = values(terms(moved, goal, edit))
        finally:
            for parameter, before in zip(parameters, origin):
                parameter.copy_(before)
    if not all(torch.equal(parameter, before) for parameter, before in zip(parameters, origin)):
        raise ValueError("The trial did not restore a fitted parameter")

    start = values(computed)
    right = {name: (displaced[1][name] - start[name]) / PARAMETER_RMS for name in start}
    central = {name: (displaced[1][name] - displaced[-1][name]) / (2 * PARAMETER_RMS) for name in start}
    return dict(losses=start, norms=norms, edit_gradient_zero=False, rms_change=PARAMETER_RMS,
                derivative=derivative, forward_difference=right, central_difference=central), output.detach()


def run(base, formal, out):
    start = monotonic()
    torch.set_num_threads(4)
    manifest = read_manifest()
    previous = checked_json(formal / "results.json", FORMAL_SHA)
    fits = checked_json(formal / "training/fits.json", FITS_SHA)
    if previous["training"] != fits["folds"]:
        raise ValueError("The training records do not match the frozen scoring")
    report = {}
    for fold in manifest["folds"]:
        family = fold["train"]
        case = load_case(base, manifest, family)
        report[family] = {}
        for label in ("pooled", "local"):
            if monotonic() - start >= 120:
                raise TimeoutError("Loss audit budget exceeded")
            engine = ConditionedEngine(label == "local")
            checkpoint = formal / "training" / f"{family}-{label}.pth"
            if digest(checkpoint) != fits["folds"][family][label]["sha256"] or engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Changed frozen head or base checkpoint")
            engine.head.load_state_dict(torch.load(checkpoint, map_location=engine.device, weights_only=True), strict=True)
            content, refs, source, goal, edit = training_tensors(engine, *case)
            parameters, output = parameter_probe(engine, content, refs, source, goal, edit)
            report[family][label] = dict(parameters=parameters, output=output_probe(output, goal, edit))
    out.mkdir()
    json_to(out / "results.json", dict(scope="frozen training-family objective audit", formal_sha256=FORMAL_SHA,
                                       checkpoint=CHECKPOINT, families=report))
    print(f"Loss audit completed in {monotonic() - start:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "formal", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
