"""Test if ordinary inputs support an additive face edit beyond exposure gains."""
from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

from probe_dast_local_face_surround import CASCADE, detector
from probe_gros_declared_edit import detect, estimated_weight
from probe_gros_local_editor_channel import actual_png, frozen_json, read_frame, request, serving
from probe_gros_local_editor_cross import REGIONS, areas_and_weight, score_outputs
from probe_physical_blue_global_fit import display, linear
from probe_playraw_mixed_light_transfer import LUMA
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.aligned_reference import fit_aligned
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


PARENT = "0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7"
QUANTILES = np.arange(1, 20) / 20


def describe(rgb, face):
    faces = detect(face, rgb)
    if len(faces) != 1:
        return None
    weight = estimated_weight(rgb, faces[0])
    y = rgb.astype(np.float64) @ LUMA
    inside = weight >= .85
    outside = (weight <= .05) & (y >= 40) & (y <= 230) & (np.ptp(rgb, axis=-1) <= 40)
    if min(inside.sum(), outside.sum()) < 384:
        return None
    face_points, background = (np.quantile(linear(rgb[mask]), QUANTILES, axis=0) for mask in (inside, outside))
    return dict(face=face_points, background=background, weight=weight)


def fit_relationship(target, reference):
    x, y = target[::2], reference[::2]
    center = x - x.mean(axis=0)
    if np.any(np.ptp(x, axis=0) < .025) or np.any(np.ptp(y, axis=0) < .025):
        return None
    gain = (x * y).sum(axis=0) / (x * x).sum(axis=0)
    slope = (center * (y - y.mean(axis=0))).sum(axis=0) / (center * center).sum(axis=0)
    offset = y.mean(axis=0) - slope * x.mean(axis=0)
    if not np.all((gain >= .25) & (gain <= 4) & (slope >= .25) & (slope <= 4)):
        return None
    test = target[1::2]
    expected = display(reference[1::2]).astype(np.int16)
    unshifted, shifted = test * gain, test * slope + offset
    if shifted.min() < -.02 or shifted.max() > 1.02:
        return None
    pure = float(np.abs(display(unshifted) - expected).mean())
    affine = float(np.abs(display(shifted) - expected).mean())
    return dict(gain=gain.tolist(), slope=slope.tolist(), offset=offset.tolist(), gain_mae=pure, affine_mae=affine,
                needs_offset=pure > 3 and affine <= .75 * pure and pure - affine >= 1)


def predict(target, reference, first, second, fallback):
    if np.array_equal(target, reference):
        return target, {"route": "exact_identity"}
    if fit_aligned(target, reference)["accepted"]:
        return fallback, {"route": "aligned"}
    if first is None or second is None:
        return fallback, {"route": "face_or_context_missing"}
    fits = {name: fit_relationship(first[name], second[name]) for name in ("face", "background")}
    result = {"models": fits}
    face, environment = fits["face"], fits["background"]
    if face is None or environment is None or environment["gain_mae"] > 3 or environment["needs_offset"]:
        return fallback, {**result, "route": "unsupported_context"}
    if face["gain_mae"] <= 3:
        return target, {**result, "route": "capture_gain_only"}
    if not face["needs_offset"]:
        return fallback, {**result, "route": "ambiguous_face"}
    delta = np.asarray(face["offset"]) / np.asarray(face["slope"])
    weight = first["weight"]
    rgb = display(linear(target) + weight[..., None] * delta)
    final = finish_spatial(target, rgb, STRENGTH)
    difference = np.abs(final[weight >= .85].astype(np.int16) - target[weight >= .85]).mean()
    if difference < 2:
        return fallback, {**result, "route": "weak_edit"}
    return final, {**result, "route": "local_offset", "target_linear_offset": delta.tolist()}


def controls(cascade, originals, out):
    scale = np.linspace(.12, .60, 19)[:, None]
    source = scale * np.array([1., .8, .65])
    gains = np.array([1.1, .85, .82])
    shifts = np.array([.1, .02, -.02])
    pure, affine = [fit_relationship(source, source * gains + bias) for bias in (0, shifts)]
    np.testing.assert_allclose(pure["offset"], np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(affine["offset"], shifts, atol=1e-9)
    rgb = originals["2"]
    same, identity = predict(rgb, rgb, None, None, np.zeros_like(rgb))
    np.testing.assert_array_equal(same, rgb)
    globals_checked = {}
    with serving(make_spatial_workbench(out / "global-control-data", enhanced=True)) as server:
        for name, image in originals.items():
            reference = np.rint(image * .83 + np.array([16., 6., 12.])).astype(np.uint8)
            own, guide = out / f"control-{name}-input.png", out / f"control-{name}-reference.png"
            save(own, image)
            save(guide, reference)
            target_id = request(server, "/api/media", own.read_bytes(), own.name)["id"]
            ref_id = request(server, "/api/media", guide.read_bytes(), guide.name)["id"]
            filename = f"control-{name}-base.png"
            baseline, baseline_sha = actual_png(server, target_id, ref_id, out / filename)
            output, reason = predict(image, reference, describe(image, cascade), describe(reference, cascade), baseline)
            fit = fit_aligned(image, reference)
            globals_checked[name] = dict(route=reason["route"], existing_alignment=fit["accepted"],
                preserved=bool(np.array_equal(output, baseline)), baseline=baseline_sha,
                trial=save(out / f"control-{name}-trial.png", output))
    grown = np.repeat(np.linspace(0, 255, 512, dtype=np.uint8)[None, :, None], 3, axis=-1)
    blank = np.repeat(grown, 300, axis=0)
    shifted = np.flip(blank, axis=1).copy()
    baseline_empty = np.rint(blank.astype(float) * .8 + 20).astype(np.uint8)
    result, no_face = predict(blank, shifted, describe(blank, cascade), describe(shifted, cascade), baseline_empty)
    np.testing.assert_array_equal(result, baseline_empty)
    return dict(identity=identity["route"], global_edits=globals_checked, no_face=no_face["route"],
                files={name: save(out / f"control-{name}.png", arr)
                       for name, arr in (("same", same), ("no-face", result))})


def run(project, output):
    cv2.setNumThreads(1)
    cascade = detector(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    source = project / "gros-local-editor-channel-v2"
    parent = checked_json(source / "results.json", PARENT)
    script = Path(__file__)
    output.mkdir()
    photos, features = {}, {}
    for name, lights in parent["intake"].items():
        for light, detail in lights.items():
            for state, sha in detail["hashes"].items():
                key = name, light, state
                photos[key] = read_frame(source / name / light / f"{state}.png", sha)
                features[key] = describe(photos[key], cascade)
    checks = controls(cascade, {name: photos[name, "cold", "off"] for name in parent["qualified"]}, output)
    outputs, decisions, files = {}, {}, {}
    for target, donor in permutations(parent["qualified"], 2):
        key = target, "cold", "off"
        slug = f"{target}-from-{donor}"
        outputs[slug], decisions[slug], files[slug] = {}, {}, {}
        for light in ("cold", "warm"):
            outputs[slug][light], decisions[slug][light] = {}, {}
            for state in ("off", "on"):
                tag = f"{light}-{state}"
                ref = donor, light, state
                baseline_name = f"spatial13-{tag}"
                baseline = read_frame(source / f"{slug}-{baseline_name}.png", parent["cases"][slug]["hashes"][baseline_name])
                graded, trace = predict(photos[key], photos[ref], features[key], features[ref], baseline)
                outputs[slug][light][state], decisions[slug][light][state] = graded, trace
                files[slug][tag] = save(output / f"{slug}-{tag}.png", graded)

    coords = frozen_json(script.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    reports = {}
    for slug, frames in outputs.items():
        target, donor = slug.split("-from-")
        original = photos[target, "cold", "off"]
        expected = read_frame(source / f"{slug}-expected.png", parent["cases"][slug]["wanted"])
        regions, _ = areas_and_weight(coords[target], coords["screen"])
        scores = score_outputs(original, expected, {"trial": frames}, regions)["trial"]
        reports[slug] = dict(scores=scores, complete_pass=all(row["passed"] for row in scores.values()),
                             traces=decisions[slug], hashes=files[slug],
                             panel=panel(output / f"{slug}-comparison.png", {
                                 "target": {"original": original, "expected": expected},
                                 "reference": {f"{light}-{state}": photos[donor, light, state]
                                               for light in ("cold", "warm") for state in ("off", "on")},
                                 "trial": {f"{light}-{state}": frame for light, pair in frames.items()
                                           for state, frame in pair.items()}}))
    json_to(output / "results.json", dict(parent=PARENT, cascade=CASCADE, opencv=cv2.__version__, controls=checks,
            cases=reports, script_sha256=digest(script), protocol_sha256=digest(script.with_name("gros-affine-intent-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
