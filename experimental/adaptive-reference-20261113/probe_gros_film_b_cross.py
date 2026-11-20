"""Cross fixed Film B with independently captured warm and cold portrait lighting."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, metrics, quantize
from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_gros_screen_chroma import ARCHIVE
from probe_target_mirror_chroma import BASE, CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze
from spatial_render import SpatialEngine, finish_spatial


FILM = "ef7a8ed92baf4109236131bbb51b3a55a9bbeaff980577f2df845d418096a30a"
LIGHTS = ("cold", "warm")


def error(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


def prepare(out, manifest, colors):
    source = BASE / "gros-portrait-light-transfer-v1" / "guides"
    view, inverse, film = colors
    portrait = {}
    identities = {}
    for person in GROUPS:
        frames = {}
        for light in LIGHTS:
            key = f"{person}-{light}"
            original = checked_rgb(source / f"{key}.png", manifest["admission"][person]["guide_sha256"][light])
            log = converted(original.astype(np.float32) / 255, inverse)
            restored = quantize(converted(log, view))
            max_error = int(np.abs(restored.astype(np.int16) - original).max())
            authored = converted(converted(log, film), view)
            clip = ((authored < 0) | (authored > 1)).any(axis=-1)
            on = quantize(np.clip(authored, 0, 1))
            effect = error(on, original)
            back = error(restored, original)
            qualified = max_error <= 1 and back <= .05 and effect >= 1
            frame_paths = {label: out / f"{key}-{label}.png" for label in ("off", "on")}
            frames[light] = {"off": original, "on": on}
            hashes = {label: save(frame_paths[label], frame) for label, frame in frames[light].items()}
            identities[key] = dict(qualified=qualified, roundtrip_mae=back, roundtrip_max=max_error,
                                   clip_fraction=float(clip.mean()), style_mae=effect, hashes=hashes)
        portrait[person] = frames
    return portrait, identities


def score(target, expected, predictions, zones):
    results = {}
    for label, mask in zones.items():
        reference = {light: metrics(target, expected, *frames.values(), mask) for light, frames in predictions.items()}
        lighting = {state: error(predictions["cold"][state][mask], predictions["warm"][state][mask])
                    for state in ("off", "on")}
        passed = (max(lighting.values()) <= 3 and
                  all(row["passed"] and row["off_matches_original_mae"] <= 3 for row in reference.values()))
        results[label] = dict(response=reference, lamp_spill=lighting, passed=passed)
    return results


def run(out):
    source = BASE / "gros-portrait-light-transfer-v1"
    manifest = checked_json(source / "results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != manifest["regions_sha256"]:
        raise ValueError("Changed registered regions")
    zones = json.loads(ledger.read_bytes())
    cube = BASE / "arri-second-film-assets-v1" / "second-film.cube"
    if digest(cube) != FILM:
        raise ValueError("Changed ARRI Film B")
    out.mkdir()
    variants, inputs = prepare(out, manifest, transforms(cube))
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed spatial model")
    cases = {}
    for person, donor in permutations(GROUPS, 2):
        target, edited = variants[person]["cold"].values()
        expected = finish_spatial(target, edited, STRENGTH)
        area = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
        guides = variants[donor]
        predictions = {key: {} for key in ("stable", "13")}
        slug = f"{person}-from-{donor}"
        source_path = out / f"{person}-cold-off.png"
        for light, frames in guides.items():
            stable, adaptive = {}, {}
            for state, guide in frames.items():
                ref = out / f"{donor}-{light}-{state}.png"
                stable[state] = apply_mapping(target, analyze(source_path, ref), STRENGTH)
                adaptive[state] = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
            predictions["stable"][light] = stable
            predictions["13"][light] = adaptive
        scores = {model: score(target, expected, states, area) for model, states in predictions.items()}
        flatten = lambda readings: {f"{light}-{state}": frame for light, group in readings.items()
                                    for state, frame in group.items()}
        compressed = {model: flatten(record) for model, record in predictions.items()}
        hashes = {f"{model}-{key}": save(out / f"{slug}-{model}-{key}.png", frame)
                  for model, output in compressed.items() for key, frame in output.items()}
        cases[slug] = dict(scores=scores, hashes=hashes,
                           expected_sha=save(out / f"{slug}-expected.png", expected),
                           panel_sha=panel(out / f"{slug}-comparison.png",
                                           {"target": {"cold": target, "Film B": expected},
                                            "reference": flatten(guides), **compressed}))
    script = Path(__file__)
    json_to(out / "results.json", dict(source=ARCHIVE, film=FILM, spatial=CHECKPOINT,
            stable=algorithm_fingerprint(), strength=STRENGTH, inputs=inputs, cases=cases,
            script_sha=digest(script), protocol_sha=digest(script.with_name("gros-film-b-cross-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
