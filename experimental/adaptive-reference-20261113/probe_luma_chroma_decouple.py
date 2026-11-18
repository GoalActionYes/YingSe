"""Try preserving the full output luminance with chroma from the fixed-luma branch."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_arri_edge_coherence import CUBE_SHA, render as apply_film
from probe_bg_real_channel_replay import histogram, infer
from probe_bg_real_editor_look import ROOT, HELD_SHA, PARENT, PARENT_SHA, evaluate, mask_for, read
from probe_dast_exposure_channels import INPUTS_SHA, fragments, safe_projection
from probe_dast_exposure_response import STATES, digest, midtone, save_report, statistics, verified_png
from spatial_render import SpatialEngine, finish_spatial

DAST_SHA = "54c69f047c40cae9db8645f0a3979cce37507f9692accde0da4f2f72705ce3ef"
EDITOR_SHA = "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d"
LUMA = np.array([.2126, .7152, .0722])


def linear(rgb):
    x = rgb / 255.
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def encoded(x):
    x = np.clip(x, 0, 1)
    return 255 * np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)


def combine(full, chroma):
    wanted = linear(full) @ LUMA
    base = linear(chroma)
    level = base @ LUMA
    ratio = np.divide(wanted, level, out=np.zeros_like(wanted), where=level > 1e-12)
    neutral = wanted[..., None]
    shift = np.where((level > 1e-12)[..., None], base * ratio[..., None] - neutral, 0.)
    positive = shift.max(axis=-1)
    factor = np.minimum(1., np.divide(1. - wanted, positive, out=np.ones_like(wanted), where=positive > 1e-12))
    pixels = np.rint(encoded(neutral + shift * factor[..., None])).astype(np.uint8)
    error = np.abs(encoded(linear(pixels) @ LUMA) - encoded(wanted))
    return pixels, {"compressed_fraction": float(np.mean(factor < 1.)),
                    "luma_error_max": float(error.max()), "luma_error_p95": float(np.percentile(error, 95))}


def renderer(engine, target):
    low, context = fragments(engine, target)

    def run(reference, baseline):
        ref = histogram(engine, reference)
        full = infer(engine, target, low, context, ref)
        if not np.array_equal(full, baseline):
            raise ValueError("Full configuration failed pixel replay")
        replace = torch.cat((context[:, :1], ref[:, 1:]), dim=1)
        hue = infer(engine, target, low, context, replace)
        return combine(full, hue)

    return run


def save(folder, name, pixels):
    path = folder / f"{name}.png"
    Image.fromarray(pixels).save(path)
    return digest(path)


def film_response(anchor, edited, original, truth, region):
    face_delta = statistics(anchor, edited, region)
    intended_face = statistics(original, truth, region)
    full = edited.astype(float) - anchor
    known = truth.astype(float) - original
    p_rgb = safe_projection(full.ravel(), known.ravel())
    p_face = (safe_projection(np.asarray(face_delta["mean_log_chroma_change"]),
                              np.asarray(intended_face["mean_log_chroma_change"]))
              if face_delta["qualified"] and intended_face["qualified"] else None)
    error = float(np.abs(full - known).mean())
    return {"face": p_face, "rgb": p_rgb, "mae": error,
            "passed": bool(p_face is not None and .75 <= p_face <= 1.25 and
                           p_rgb is not None and .75 <= p_rgb <= 1.25 and error <= 3)}


def dast(engine, prep, replay, cube, out):
    plan = json.loads((prep / "inputs.json").read_bytes())
    old = json.loads((replay / "results.json").read_bytes())
    records = []
    for index, group in enumerate(plan["groups"], 1):
        saved = old["groups"][index - 1]["hashes"]
        target = plan["groups"][group["target_group"] - 1]
        dest = out / f"portrait-{index}"
        dest.mkdir()
        target_folder = prep / str(target["group"])
        rgb = verified_png(target_folder / "nominal.png", target["frames"]["nominal"])
        mask = verified_png(target_folder / "nominal-mask.png", target["target_mask_sha256"]) > 0
        refs = {key: verified_png(prep / str(index) / f"{key}.png", group["frames"][key]) for key in STATES}
        refs["film_a"] = apply_film(refs["nominal"], cube)
        truth = finish_spatial(rgb, apply_film(rgb, cube), .8)
        transform = renderer(engine, rgb)
        before, after, artifacts, diagnostics = {}, {}, {}, {}
        for key, ref in refs.items():
            name = key if key == "nominal" else f"{key}-full"
            before[key] = verified_png(replay / str(index) / f"{name}.png", saved[name])
            after[key], diagnostics[key] = transform(ref, before[key])
            artifacts[key] = save(dest, key, after[key])
        scores = {}
        look = {}
        for key in (*STATES[1:], "film_a"):
            area = mask & midtone(rgb) & midtone(before["nominal"]) & midtone(after["nominal"])
            area &= midtone(before[key]) & midtone(after[key])
            if key == "film_a":
                area &= midtone(truth)
                look = {arm: film_response(values["nominal"], values[key], rgb, truth, area)
                        for arm, values in (("baseline", before), ("candidate", after))}
            else:
                old_stats = statistics(before["nominal"], before[key], area)
                new_stats = statistics(after["nominal"], after[key], area)
                scores[key] = {"pixels": old_stats["pixels"], "baseline": old_stats.get("chroma_norm"),
                               "candidate": new_stats.get("chroma_norm")}
        records.append({"group": index, "exposure": {key: scores[key] for key in STATES[1:]},
                        "film_a": look, "hashes": artifacts, "render": diagnostics})
    return records


def background(engine, folder, out):
    source = ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    if (digest(source), digest(PARENT / "results.json")) != (HELD_SHA, PARENT_SHA):
        raise ValueError("Background groups changed")
    old = json.loads((folder / "results.json").read_bytes())
    targets = {row["pair"]: row for row in json.loads((PARENT / "results.json").read_bytes())["cases"]}
    cases = {}
    for key, detail in old["cases"].items():
        target = read(targets[key]["input_file"])
        person = mask_for(targets[key], key.split("-")[0])
        path = folder / key
        known = {"off": target, "on": verified_png(path / "expected-on.png", detail["artifacts"]["expected-on.png"])}
        apply = renderer(engine, target)
        dest = out / key
        dest.mkdir()
        baseline, candidate, artifacts, diagnostics = {}, {}, {}, {}
        for role in ("C1", "F1"):
            baseline[role], candidate[role] = {}, {}
            for state in ("off", "on"):
                key_base = f"{role}-{state}"
                ref, original = [verified_png(path / f"{key_base}-{suffix}.png",
                                  detail["artifacts"][f"{key_base}-{suffix}.png"]) for suffix in ("reference", "13")]
                baseline[role][state] = original
                candidate[role][state], diagnostics[key_base] = apply(ref, original)
                artifacts[key_base] = save(dest, key_base, candidate[role][state])
        known_delta = known["on"].astype(float) - target
        errors = {arm: {role: float(np.abs(group["on"].astype(float) - group["off"] - known_delta).mean())
                        for role, group in images.items()} for arm, images in (("baseline", baseline), ("candidate", candidate))}
        cases[key] = {"baseline": evaluate(target, known, baseline, person), "candidate": evaluate(
            target, known, candidate, person), "response_rgb_mae": errors, "hashes": artifacts, "render": diagnostics}
    return cases


@torch.inference_mode()
def run(prep, replay, cube_path, editor, out):
    actual = (digest(prep / "inputs.json"), digest(replay / "results.json"),
              digest(cube_path), digest(editor / "results.json"))
    if actual != (INPUTS_SHA, DAST_SHA, CUBE_SHA, EDITOR_SHA):
        raise ValueError("One of the registered comparison inputs changed")
    cube = np.loadtxt(cube_path, skiprows=4, dtype=np.float32).reshape(64, 64, 64, 3).transpose(2, 1, 0, 3)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    out.mkdir()
    portraits = dast(engine, prep, replay, cube, out)
    swaps = background(engine, editor, out)
    exposure = [row["exposure"][key] for row in portraits for key in STATES[2:]]
    eligible = [row for row in exposure if row["baseline"] is not None]
    gains = sum(row["candidate"] <= .75 * row["baseline"] for row in eligible)
    portrait_increase = max(row["candidate"] - row["baseline"] for row in eligible)
    film_passes = {arm: sum(row["film_a"][arm]["passed"] for row in portraits) for arm in ("baseline", "candidate")}
    swap_passes = {arm: sum(row[arm]["passed"] for row in swaps.values()) for arm in ("baseline", "candidate")}
    worst_swap = max(row["candidate"]["max_protected_swap"] for row in swaps.values())
    regressions = [row["film_a"]["candidate"]["mae"] - row["film_a"]["baseline"]["mae"] for row in portraits]
    regressions.extend(row["response_rgb_mae"]["candidate"][role] - row["response_rgb_mae"]["baseline"][role]
                       for row in swaps.values() for role in ("C1", "F1"))
    diagnostics = [values for row in [*portraits, *swaps.values()] for values in row["render"].values()]
    worst_luma = max(row["luma_error_p95"] for row in diagnostics)
    # The same-reference chroma condition is unchanged if both input photographs are identical.
    gray = np.tile(np.arange(256, dtype=np.uint8), (3, 1))[None].transpose(0, 2, 1)
    identity, _ = combine(gray, gray)
    np.testing.assert_array_equal(identity, gray)
    first_frame = json.loads((prep / "inputs.json").read_bytes())["groups"][0]["frames"]["nominal"]
    same = verified_png(prep / "1/nominal.png", first_frame)
    unguarded = finish_spatial(same, engine.prepare(same, same).apply(same), .8)
    recombined, _ = renderer(engine, same)(same, unguarded)
    np.testing.assert_array_equal(unguarded, recombined)
    passed = (len(eligible) == 20 and gains >= 16 and portrait_increase <= .03 and worst_luma <= 1
              and film_passes["candidate"] >= 3 and swap_passes["candidate"] >= 3
              and worst_swap <= 3 and max(regressions) <= 2)
    save_report(out / "results.json", {"input_hashes": actual, "weight_sha256": engine.checkpoint_sha,
                "exposure_eligible": len(eligible), "exposure_improved": gains, "max_exposure_increase": portrait_increase,
                "film_passes": film_passes, "bg_real_passes": swap_passes, "bg_real_worst_swap": worst_swap,
                "worst_intentional_rgb_regression": max(regressions), "worst_luma_error_p95": worst_luma,
                "images": len(diagnostics), "passed": passed, "dast": portraits, "background": swaps})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "replay", "lut", "editor", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.prepared, args.replay, args.lut, args.editor, args.output)
