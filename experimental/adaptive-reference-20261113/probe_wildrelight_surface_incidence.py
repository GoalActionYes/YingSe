"""Probe light-direction weighting on fixed outdoor reference pairs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_capture_invariant_look import render_style, style_transforms
from probe_gros_film_b_cross import score
from probe_gros_render_intent_projection import grade
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)
from probe_wildrelight_measured_gain import INPUT_SHA, PARENT, REGIONS, areas_for
from probe_wildrelight_wall_cross import SCENES, TIMES, image, prepared_rgb

MEASURED = "aa61abe70f5b12fa5fb9eb52f595a56cfef19299ef71fbaf6fe828fac67e2332"
SURFACES = {"wall": "vertical_azimuth_average", "pillar": "upward"}


def integrate(panorama, surface):
    theta = np.pi * (np.arange(len(panorama)) + .5) / len(panorama)
    area = np.sin(theta)
    if surface == "vertical_azimuth_average":
        incidence = area / np.pi
    elif surface == "upward":
        incidence = np.maximum(np.cos(theta), 0)
    else:
        incidence = np.ones_like(area)
    return np.average(panorama.mean(1), weights=area * incidence, axis=0)


def check():
    uniform = np.broadcast_to([.25, .5, .75], (8, 16, 3))
    for direction in (*SURFACES.values(), "sphere"):
        np.testing.assert_allclose(integrate(uniform, direction), [.25, .5, .75], atol=1e-12, rtol=0)
    lower = np.repeat((np.arange(8) >= 4)[:, None, None], 3, axis=2)
    assert np.array_equal(integrate(lower, "upward"), np.zeros(3))
    np.testing.assert_allclose(integrate(lower, "vertical_azimuth_average"), .5, atol=1e-12, rtol=0)


def load_capture(raw, files, scene, time, kind):
    filename = f"{time}_{'hdr' if kind == 'photo' else 'envmap'}.exr"
    key = f"small-aligned/{scene}/{kind}/{filename}"
    path = raw / key
    if digest(path) != files["inputs"][key]:
        raise ValueError(f"Unexpected source {key}")
    return image(path)


def measure_ref(scene, raw, files, prior, labels, film, out):
    record = files["scenes"][scene]
    images = {kind: {time: load_capture(raw, files, scene, time, kind) for time in TIMES}
              for kind in ("photo", "envmap")}
    actual, environment = images["photo"], images["envmap"]
    scale = record["scale"]
    earlier, later = (prepared_rgb(actual[t], scale) for t in TIMES)
    roi = areas_for(earlier.shape[:2], labels[scene]["neutral"])
    normal = SURFACES[scene]
    means = {name: {time: integrate(env, name) for time, env in environment.items()}
             for name in ("sphere", normal)}
    for time in TIMES:
        np.testing.assert_allclose(means["sphere"][time], record["envmaps"][time]["sphere_mean_rgb"], atol=1e-12, rtol=0)
    gain = means[normal]["time0"] / means[normal]["time5"]
    if not np.isfinite(gain).all() or not (gain > 0).all():
        raise ValueError("Invalid directional gain")
    candidate = prepared_rgb(actual["time5"] * gain, scale)
    audited = {"off": candidate, "on": render_style(candidate, film)}

    y, x = np.ogrid[:earlier.shape[0], :earlier.shape[1]]
    train = ((y // 16 + x // 16) % 2 == 0) & roi["neutral"]
    held = roi["neutral"] & ~train
    left, right = (actual[time][train] for time in TIMES)
    fitted = (left * right).sum(axis=0) / (right * right).sum(axis=0)
    sphere = prepared_rgb(actual["time5"] * prior["gains"]["rgb"], scale)
    np.testing.assert_array_equal(sphere, checked_rgb(
        BASE / "wildrelight-measured-gain-v1" / f"{scene}-reference-rgb-off.png", prior["hashes"]["rgb-off"]))
    arms = {"base": later, "sphere": sphere, "surface": candidate,
            "paired_reference_only": prepared_rgb(actual["time5"] * fitted, scale)}
    metrics = {key: {label: float(np.abs(earlier[mask].astype(float) - result[mask]).mean())
                    for label, mask in ({"held": held} if key == "paired_reference_only" else (roi | {"held": held})).items()}
               for key, result in arms.items()}
    images = {key: save(out / f"{scene}-reference-{key}.png", frame) for key, frame in audited.items()}
    board = panel(out / f"{scene}-reference-comparison.png", {
        "captured": {"early": earlier, "late": later},
        "late corrected": {"whole sphere": sphere, "surface orientation": candidate},
    })
    return audited, dict(surface=normal, gains=gain.tolist(), reference_mae=metrics,
                         pixels=[int(train.sum()), int(held.sum())], paired_fit=fitted.tolist(), hashes=images, board=board)


def run(out):
    root = BASE / "wildrelight-small-v1"
    prep = BASE / "wildrelight-wall-prepared-v1"
    old = BASE / "wildrelight-wall-cross-v1"
    files = checked_json(prep / "inputs.json", INPUT_SHA)
    cross = checked_json(old / "results.json", PARENT)
    measured = checked_json(BASE / "wildrelight-measured-gain-v1/results.json", MEASURED)
    labels = checked_json(Path(__file__).with_name("wildrelight-wall-cross-regions.json"), REGIONS)
    check()
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or cross["strength"] != STRENGTH:
        raise ValueError("Changed renderer")
    film = style_transforms(BASE)["film_b"]
    out.mkdir()
    references, report = {}, {}
    for scene in SCENES:
        references[scene], report[scene] = measure_ref(scene, root, files, measured["references"][scene], labels, film, out)
    results = {}
    for scene, donor in (SCENES, SCENES[::-1]):
        target = checked_rgb(prep / f"{scene}-time0.png", files["scenes"][scene]["hashes"]["time0"])
        expected = finish_spatial(target, render_style(target, film), STRENGTH)
        zone = areas_for(target.shape[:2], labels[scene]["neutral"])
        earlier = {state: checked_rgb(old / f"{scene}-from-{donor}-time0-{state}.png",
                                      cross["cases"][scene]["hashes"][f"time0-{state}"]) for state in ("off", "on")}
        sphere = {state: checked_rgb(BASE / "wildrelight-measured-gain-v1" / f"{scene}-from-{donor}-rgb-{state}.png",
                                      measured["cases"][scene]["hashes"][f"rgb-{state}"]) for state in ("off", "on")}
        if score(target, expected, {"cold": earlier, "warm": sphere}, zone) != measured["cases"][scene]["scores"]["rgb"]:
            raise ValueError("Sphere comparison changed")
        candidate = {role: grade(engine, target, rgb) for role, rgb in references[donor].items()}
        results[scene] = dict(
            sphere=measured["cases"][scene]["scores"]["rgb"],
            surface=score(target, expected, {"cold": earlier, "warm": candidate}, zone),
            hashes={role: save(out / f"{scene}-grade-{role}.png", png) for role, png in candidate.items()},
            board=panel(out / f"{scene}-grade-comparison.png", {
                "target": {"original": target, "expected": expected},
                "early": earlier, "sphere": sphere, "surface": candidate,
            }),
        )
    protocol = Path(__file__).with_name("wildrelight-surface-incidence-protocol.md")
    json_to(out / "results.json", dict(parent=MEASURED, inputs=INPUT_SHA, regions=REGIONS,
                                      checkpoint=CHECKPOINT, strength=STRENGTH, protocol=digest(protocol),
                                      references=report, cases=results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
