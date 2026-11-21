"""Trace real measured lamp deltas through a frozen nonlinear grading operation."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import torch

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, metrics, quantize
from probe_gros_film_b_cross import FILM, score
from probe_multilum_gray_probe import linear
from probe_synclight_real_pixels import GAIN, LUMA, shot, srgb8
from probe_target_mirror_chroma import BASE, CHECKPOINT, checked_rgb, digest, save
from spatial_render import SpatialEngine, finish_spatial


SCENES = ("scene35", "scene25")


def regions(base, fraction):
    return {
        "whole": np.ones(fraction.shape, dtype=bool),
        "weak": fraction <= .10,
        "strong": fraction >= .30,
        "neutral": (np.ptp(base, axis=-1) <= 20) & (base @ LUMA >= 40) & (base @ LUMA <= 220),
    }


def adjusted(cold, delta_code, delta_linear):
    out8 = srgb8(linear(cold) - delta_linear)
    code = np.clip(cold.astype(np.int16) - delta_code, 0, 255).astype(np.uint8)
    return {"subtract_code": code, "subtract_linear": out8}


def render_state(rawpy, source, ledger, anatomy, previous, scene, camera, style):
    exposures = previous["records"][scene]
    files = [source / f"{scene}_{camera}_{key}.DNG" for key in ("0_ambient", "1_camvis")]
    for path in files:
        assert digest(path) == ledger["files"][path.name]["sha256"]
    ambient, lamp = [shot(rawpy, path, anatomy[scene]["white_balance"])[0] for path in files]
    amount = previous["ambient_multiplier"]
    exposure = exposures["exposure"]
    standard = (ambient * amount + lamp) * exposure
    blue = (ambient * amount + lamp * GAIN) * exposure
    lamp_y, full_y = lamp @ LUMA, (ambient * amount + lamp) @ LUMA
    fraction = np.divide(lamp_y, full_y, out=np.zeros_like(lamp_y), where=full_y > 0)
    originals = {"std": srgb8(standard), "cold": srgb8(blue)}
    source_arm = BASE / "synclight-ambient-balance" / scene
    for key, expected in (("std", "base"), ("cold", "cold")):
        path = source_arm / f"camera{camera}-{expected}.png"
        witness = exposures["output_sha256"][path.name]
        np.testing.assert_array_equal(originals[key], checked_rgb(path, witness))
    previous_fraction_hash = exposures["fraction_sha256"]
    if camera == 2:
        old = source_arm / "lamp-fraction.npy"
        assert digest(old) == previous_fraction_hash
        np.testing.assert_allclose(fraction, np.load(old, allow_pickle=False), rtol=0, atol=1e-12)
    styled = {key: style(pixel) for key, pixel in originals.items()}
    delta_code = originals["cold"].astype(np.int16) - originals["std"]
    delta_light = blue - standard
    arms = {
        "std": {"off": originals["std"], "on": styled["std"]},
        "cold": {"off": originals["cold"], "on": styled["cold"]},
    }
    after_off = adjusted(originals["cold"], delta_code, delta_light)
    after_on = adjusted(styled["cold"], delta_code, delta_light)
    for name in after_off:
        arms[name] = {"off": after_off[name], "on": after_on[name]}
    safe = np.logical_and.reduce([((state >= 16) & (state <= 239)).all(axis=2)
                                  for state in (*originals.values(), *styled.values())])
    return arms, regions(originals["std"], fraction), safe


def pair_reference(arms, areas, safe):
    output = {}
    for name, pair in arms.items():
        areas_checked = {key: mask & safe for key, mask in areas.items()}
        data = {}
        for key, mask in areas_checked.items():
            count = int(mask.sum())
            if count < 256:
                data[key] = {"pixels": count, "qualified": False}
                continue
            original, expected = arms["std"].values()
            row = metrics(original, expected, pair["off"], pair["on"], mask)
            data[key] = row | {"pixels": count, "qualified": True,
                               "absolute_on_passed": row["on_matches_authored_mae"] <= 3}
        output[name] = data
    return output


@torch.inference_mode()
def run(decoder, raw, out):
    sys.path.insert(0, str(decoder.resolve()))
    import rawpy

    torch.set_num_threads(4)
    assert rawpy.__version__ == "0.28.0"
    witness = json.loads(Path("docs/research/2026-11-18-synclight-real-pixel-evidence.json").read_bytes())
    assert digest(raw / "intake.json") == witness["intake_sha256"]
    ledger = json.loads((raw / "intake.json").read_bytes())
    balances = BASE / "synclight-ambient-balance/results.json"
    previous = json.loads(balances.read_bytes())
    assert previous["ambient_multiplier"] == 11
    cube = BASE / "arri-second-film-assets-v1" / "second-film.cube"
    assert digest(cube) == FILM
    view, inverse, film = transforms(cube)

    def style(frame):
        codes = frame.astype(np.float32) / 255
        return quantize(np.clip(converted(converted(converted(codes, inverse), film), view), 0, 1))

    engine = SpatialEngine("multiref-v1")
    assert engine.checkpoint_sha == CHECKPOINT
    out.mkdir()
    records = {}
    for scene in SCENES:
        reference, references_areas, ref_safe = render_state(rawpy, raw, ledger, witness["records"], previous, scene, 1, style)
        target_states, target_areas, target_safe = render_state(rawpy, raw, ledger, witness["records"], previous, scene, 2, style)
        target, authored = target_states["std"].values()
        expected = finish_spatial(target, authored, .8)
        folder = out / scene
        folder.mkdir()
        reference_scores = pair_reference(reference, references_areas, ref_safe)
        records[scene] = {"reference": reference_scores, "reference_safe_fraction": float(ref_safe.mean()),
                          "target_safe_fraction": float(target_safe.mean()), "target": {}, "hashes": {}}
        row = records[scene]
        scored_regions = {name: region for name, region in target_areas.items() if region.sum() >= 256}
        predictions = {}
        for name, pair in reference.items():
            predictions[name] = {}
            for state, guide in pair.items():
                result = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                predictions[name][state] = result
                for label, frame in (("ref", guide), ("output", result)):
                    file = f"{name}-{state}-{label}.png"
                    row["hashes"][file] = save(folder / file, frame)
        for name, pair in predictions.items():
            row["target"][name] = score(target, expected, {"cold": predictions["std"], "warm": pair}, scored_regions)
        print(scene, {name: {k: v["response"]["warm"]["projection"] for k, v in region.items()}
                      for name, region in row["target"].items()}, flush=True)
    result = {"records": records, "base_sha256": digest(balances), "film_sha256": FILM,
              "source_sha256": digest(Path(__file__))}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("decoder", "raw", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
