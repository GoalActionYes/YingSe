"""Recombine full luminance and chroma from two frozen portrait renderers."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_ppst_first_portrait import gradient_correlation
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, panel, save

SCENE = "01-from-02"
DETAIL_SHA = "05f3ba83b5a2245df1d7fdab3dc26f72537a4c54d37a731dcdb4dd35488612de"
LUMA = np.array([.2126, .7152, .0722])
STATES = ("off", "on")


def luma(image):
    return image @ LUMA


def swap(luma_source, chroma_source):
    output = chroma_source.astype(np.float64) + (
        luma(luma_source) - luma(chroma_source)
    )[..., None]
    clip = ((output < 0) | (output > 255)).mean()
    return np.rint(np.clip(output, 0, 255)).astype(np.uint8), float(clip)


def run(out):
    photos = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(photos / "results.json", BASELINE_SHA)
    row = manifest["cases"][SCENE]
    detail = BASE / "ppst-detail-render-v1"
    frozen = checked_json(detail / "results.json", DETAIL_SHA)["methods"]["guided_residual"]
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    original = checked_rgb(photos / "01-cold-off.png", manifest["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(photos / f"{SCENE}-expected.png", row["expected_sha"])
    regions = masks(labels["01"]["cold"], labels["screen"], original.shape[:2])
    baseline = {"13": {}, "PPST": {}}
    for light in LIGHTS:
        baseline["13"][light], baseline["PPST"][light] = {}, {}
        for state in STATES:
            key = f"{light}-{state}"
            baseline["13"][light][state] = checked_rgb(photos / f"{SCENE}-13-{key}.png", row["hashes"][f"13-{key}"])
            baseline["PPST"][light][state] = checked_rgb(
                detail / f"guided_residual-{key}.png", frozen["media"][key]["image_sha256"])
    results = {arm: score(original, expected, renders, regions) for arm, renders in baseline.items()}
    if results["13"] != row["scores"]["13"] or results["PPST"] != frozen["scores"]:
        raise ValueError("Frozen grades did not reproduce")

    out.mkdir()
    output, media = {}, {}
    for tone_arm, color_arm in (("13", "PPST"), ("PPST", "13")):
        name = f"{tone_arm}-luma-{color_arm}-chroma"
        output[name], media[name] = {}, {}
        for light in LIGHTS:
            output[name][light] = {}
            for state in STATES:
                tone = baseline[tone_arm][light][state]
                chroma = baseline[color_arm][light][state]
                np.testing.assert_array_equal(swap(tone, tone)[0], tone)
                np.testing.assert_array_equal(swap(chroma, chroma)[0], chroma)
                frame, clip = swap(tone, chroma)
                output[name][light][state] = frame
                key = f"{light}-{state}"
                media[name][key] = dict(
                    sha256=save(out / f"{name}-{key}.png", frame),
                    clip_fraction=clip,
                    structure=gradient_correlation(original, frame),
                )
        results[name] = score(original, expected, output[name], regions)

    boards = {}
    for light in LIGHTS:
        rows = {"target": {"original": original, "Film B": expected}}
        rows.update({name: refs[light] for name, refs in (baseline | output).items()})
        boards[light] = panel(out / f"{light}-comparison.png", rows)
    json_to(out / "results.json", dict(parent=BASELINE_SHA, detail=DETAIL_SHA, scores=results,
                                       media=media, boards=boards))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
