"""Keep only the part of a frozen lighting correction that depends on position."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_gros_dinolight_reference import mae, resized
from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, finish_spatial,
    json_to, panel, save,
)

PREVIOUS = "ccbe004194672eb412e8e3846cc755389dc5cadc005adc6be78a56a42afa47c8"
CL3AN_HASH = "2ac743dd76c29035a18ceaf37d7a8c7eecd3e511ea987d146517428f6dd95d30"
NOISE = 1e-12


def basis(image):
    r, g, b = np.moveaxis(image, -1, 0)
    return np.stack((np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b), axis=-1)


def spatial_only(source, corrected):
    original = source.astype(np.float64) / 255
    delta = (corrected.astype(np.float64) - source) / 255
    features = basis(original)
    sample = features[::4, ::4]
    y, x = np.ogrid[:sample.shape[0], :sample.shape[1]]
    use = (x + y) % 2 == 0
    coef, _, rank, _ = np.linalg.lstsq(sample[use], delta[::4, ::4][use], rcond=None)
    residual = delta - features @ coef
    estimate = original + residual
    output = np.rint(np.clip(estimate, 0, 1) * 255).astype(np.uint8)
    notes = dict(rank=int(rank), fitted_delta_rgb8=float(np.abs(delta-residual).mean() * 255),
                 residual_rgb8=float(np.abs(residual).mean() * 255),
                 clip_fraction=float(((estimate < -NOISE) | (estimate > 1 + NOISE)).mean()))
    return output, notes


def check_controls():
    source = np.random.default_rng(38).integers(0, 201, (32, 64, 3), dtype=np.uint8)
    expected, _ = spatial_only(source, source)
    np.testing.assert_array_equal(expected, source)
    for channel in range(3):
        source[..., channel] = (source[..., channel] // 32) * 32
    known = source.astype(np.int16)
    known[..., 0] += (known[..., 1] // 32) ** 2
    known[..., 1] += (source[..., 0] // 32) * (source[..., 2] // 32)
    expected, _ = spatial_only(source, known.astype(np.uint8))
    np.testing.assert_array_equal(expected, source)


def run(out):
    check_controls()
    photos = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(photos / "results.json", BASELINE_SHA)
    annotation = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    prior = BASE / "gros-dinolight-reference-v2"
    frozen = checked_json(prior / "results.json", PREVIOUS)
    cl3an = BASE / "cl3an-real-light-input"
    source_ids = checked_json(BASE.parents[1] / "docs/research/2026-11-21-cl3an-pointwise-bound-evidence.json",
                              CL3AN_HASH)

    out.mkdir()
    physical = {}
    for scene, evidence in source_ids.items():
        original = resized(checked_rgb(cl3an / f"{scene}_1_IN.png", evidence["source_sha256"]["color"]))
        ambient = resized(checked_rgb(cl3an / f"{scene}_GT.png", evidence["source_sha256"]["ambient"]))
        restored = checked_rgb(prior / f"cl3an-{scene}-restored.png", frozen["physical"][scene]["sha"])
        candidate, notes = spatial_only(original, restored)
        before, after = mae(original, ambient), mae(candidate, ambient)
        physical[scene] = dict(before=before, full=mae(restored, ambient), after=after, passed=after <= .8*before,
                               notes=notes, sha=save(out / f"cl3an-{scene}-spatial.png", candidate))

    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or manifest["strength"] != STRENGTH:
        raise ValueError("Changed grade engine or strength")
    subject = checked_rgb(photos / "01-cold-off.png", manifest["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(photos / "01-from-02-expected.png", manifest["cases"]["01-from-02"]["expected_sha"])
    zones = masks(annotation["01"]["cold"], annotation["screen"], subject.shape[:2])
    refs, grades = {}, {}
    media, boards = {}, {}
    for light in LIGHTS:
        key = f"02-{light}"
        refs[light], grades[light] = {}, {}
        media[light] = {}
        for state in ("off", "on"):
            source = resized(checked_rgb(photos / f"{key}-{state}.png", manifest["inputs"][key]["hashes"][state]))
            restored = checked_rgb(prior / f"ref-{light}-{state}.png", frozen["references"][light][state]["sha"])
            image, notes = spatial_only(source, restored)
            grade = finish_spatial(subject, engine.prepare(subject, image).apply(subject), STRENGTH)
            refs[light][state], grades[light][state] = image, grade
            media[light][state] = dict(notes=notes, reference_shift=mae(source, image),
                                       ref=save(out / f"ref-{light}-{state}.png", image),
                                       grade=save(out / f"grade-{light}-{state}.png", grade))
        media[light]["reference_edit_mae"] = mae(refs[light]["off"], refs[light]["on"])
        boards[light] = panel(out / f"{light}-comparison.png", {
            "target": {"original": subject, "Film B": expected},
            "scope": refs[light], "spatial correction": grades[light],
            **{arm: {state: checked_rgb(prior / f"{arm}-{light}-{state}.png",
                                       frozen["outputs"][arm][f"{light}-{state}"]) for state in ("off", "on")}
               for arm in ("resize", "DINOLight")},
        })
    ref_change = {state: mae(refs["cold"][state], refs["warm"][state]) for state in ("off", "on")}
    json_to(out / "results.json", dict(previous=PREVIOUS, checkpoint=CHECKPOINT, physical=physical,
                                       media=media, boards=boards, reference_lamp_delta=ref_change,
                                       scores=score(subject, expected, grades, zones)))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
