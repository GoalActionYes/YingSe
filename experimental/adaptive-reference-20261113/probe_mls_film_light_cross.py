"""Cross an actual reference illuminant change with one fixed digital edit."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import LOOK_SHA, converted, digest, metrics, quantize
from probe_arri_sdr_renderability import transforms
from probe_hdm_chart import CHECKPOINT
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from spatial_render import SpatialEngine, finish_spatial


SOURCE_SHA = "197a0864b05c83c0b3c2aee346d59192784104594b658460c5b768ff5fd1c2a4"


def run(base, out):
    source = base / "mls-linear-pilot"
    frozen = verified_json(source / "results.json", SOURCE_SHA)
    lut = base / "arri-shared-look-assets-v1/film-a.cube"
    if digest(lut) != LOOK_SHA:
        raise ValueError("Changed Film A LUT")
    view, inverse, film = transforms(lut)
    out.mkdir()
    scenes, prepared = {}, {}
    for scene in ("02", "03"):
        scenes[scene], prepared[scene] = {}, {}
        for light, suffix in (("standard", ""), ("blue", "_LED-B025")):
            filename = f"{scene}_2HAL_DESK{suffix}.png"
            raw = verified(source / filename, frozen["scenes"][scene]["inputs_sha256"][filename])
            rgb = raw.astype(np.float32) / 255
            log = converted(rgb, inverse)
            np.testing.assert_array_equal(raw, quantize(converted(log, view)))
            edited = quantize(np.clip(converted(converted(log, film), view), 0, 1))
            path = out / f"{scene}-{light}-reference-film.png"
            Image.fromarray(edited).save(path)
            scenes[scene][light] = {"off": raw, "on": edited}
            prepared[scene][light] = digest(path)

    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 checkpoint")
    cases = {}
    for scene, donor in (("02", "03"), ("03", "02")):
        target, authored = scenes[scene]["standard"].values()
        expected = finish_spatial(target, authored, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in (target, authored)))
        if neutral.sum() < 256:
            raise ValueError(f"Insufficient common neutral support: {scene}")
        outputs, hashes, response = {}, {}, {}
        for light, refs in scenes[donor].items():
            outputs[light], hashes[light] = {}, {}
            for state, guide in refs.items():
                result = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                path = out / f"{scene}-from-{donor}-{light}-{state}.png"
                Image.fromarray(result).save(path)
                hashes[light][state] = digest(path)
                outputs[light][state] = result
            response[light] = {region: metrics(target, expected, *outputs[light].values(), mask)
                               for region, mask in (("whole", whole), ("neutral", neutral))}
        spill = {
            state: float(np.abs(outputs["standard"][state][neutral].astype(float)
                                - outputs["blue"][state][neutral]).mean())
            for state in ("off", "on")
        }
        cases[scene] = {"reference_scene": donor, "outputs_sha256": hashes, "response": response,
                        "neutral_pixels": int(neutral.sum()), "neutral_lamp_spill_mae": spill,
                        "passed": max(spill.values()) <= 3 and
                        all(value["passed"] for group in response.values() for value in group.values())}
        board = Image.new("RGB", (1080, 492), "#222730")
        draw = ImageDraw.Draw(board)
        tables = (scenes[donor], outputs)
        for row, table in enumerate(tables):
            for col, (light, state) in enumerate((("standard", "off"), ("standard", "on"), ("blue", "off"), ("blue", "on"))):
                tile = Image.fromarray(table[light][state])
                tile.thumbnail((256, 144))
                x, y = col * 270 + 6, row * 164
                draw.text((x, y + 4), f"{('reference', '13 (80%)')[row]} / {light} / {state}", fill="white")
                board.paste(tile, (x, y + 20))
        for col, (label, rgb) in enumerate((("target", target), ("known Film A", expected))):
            tile = Image.fromarray(rgb)
            tile.thumbnail((256, 144))
            x = col * 270 + 6
            draw.text((x, 332), label, fill="white")
            board.paste(tile, (x, 348))
        panel = out / f"{scene}-comparison.png"
        board.save(panel)
        cases[scene]["comparison_sha256"] = digest(panel)
    json_to(out / "results.json", {"source_sha256": SOURCE_SHA, "lut_sha256": LOOK_SHA,
                                   "weight_sha256": CHECKPOINT, "references": prepared,
                                   "cases": cases, "passed": all(case["passed"] for case in cases.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
