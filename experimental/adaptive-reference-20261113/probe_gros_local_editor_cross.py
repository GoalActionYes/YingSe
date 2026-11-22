"""Cross one recorded local-editor operation with new matched portrait capture pairs."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from probe_gros_film_b_cross import error, score
from probe_playraw_mixed_light_cross_editor import default_states
from probe_playraw_mixed_light_skin import INPUT_HASHES, read
from probe_playraw_mixed_light_transfer import chroma
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze
from spatial_render import SpatialEngine, finish_spatial
from triplet_color import to_srgb


CAPTURES = "45c4a6449be1b7f1b4e0a43b688fd737a010d18a63413193d7f2be9a11c810ce"
REGIONS = "0f8508e2b6d195d5c42a5e145590435d3c9f4247883170cf52bf3679ccdad88a"
SIZE = (1024, 683)
LIGHTS = ("cold", "warm")
STATES = ("off", "on")


def export_states(cli, source, settings, folder, *, floating=False):
    profile, extension, format_conf = (
        ("lin_rec709", "exr", "plugins/imageio/format/exr/bpp=32") if floating
        else ("srgb", "tif", "plugins/imageio/format/tiff/bpp=8")
    )
    editor_env = {key: value for key, value in os.environ.items() if key != "OMP_NUM_THREADS"}
    exports = {}
    for state, xmp in zip(STATES, settings, strict=True):
        sidecar = folder / f"{state}.xmp"
        sidecar.write_bytes(xmp)
        config, cache = folder / f"config-{state}", folder / f"cache-{state}"
        config.mkdir()
        cache.mkdir()
        dest = folder / f"editor-{state}.{extension}"
        args = [path.as_posix() for path in (cli, source, sidecar, dest)] + [
            "--width", "1024", "--height", "1024", "--hq", "true", "--apply-custom-presets", "false",
            "--icc-type", profile, "--export_masks", "false", "--core", "--configdir", config.as_posix(),
            "--cachedir", cache.as_posix(), "--disable-opencl", "-t", "2", "--conf", "write_sidecar_files=never",
            "--conf", format_conf, "--conf", "plugins/darkroom/sharpen/auto_apply=false",
        ]
        with (folder / f"{state}.log").open("wb") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True,
                           cwd=cli.parent, env=editor_env)
        exports[state] = dest
    return exports


def areas_and_weight(spec, screen):
    y, x = np.ogrid[:SIZE[1], :SIZE[0]]
    x, y = x * (384 / SIZE[0]), y * (256 / SIZE[1])
    cx, cy, rx, ry = spec["ellipse"]
    distance = np.hypot((x - cx) / rx, (y - cy) / ry)
    t = np.clip((distance - .70) / .30, 0, 1)
    alpha = 1 - t * t * (3 - 2 * t)
    regions = {"whole": np.ones(alpha.shape, dtype=bool)}
    for name, (x0, y0, x1, y1) in (screen | {k: spec[k] for k in ("cheek_left", "cheek_right")}).items():
        regions[name] = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return regions, alpha


def prepare_frame(original, identity, settings, cli, folder, regions, alpha):
    if digest(original) != identity:
        raise ValueError(f"Changed original capture: {original}")
    folder.mkdir(parents=True)
    with Image.open(original) as image:
        rgb, policy = to_srgb(image)
        sized = rgb.resize(SIZE, Image.Resampling.LANCZOS)
        sized.save(folder / "input.tif", icc_profile=rgb.info["icc_profile"])
        before = np.array(sized)
    exports = export_states(cli, folder / "input.tif", settings, folder)
    edited = {state: read(path, False)[0] for state, path in exports.items()}
    if any(image.shape != before.shape for image in edited.values()):
        raise ValueError("Editor changed image dimensions")
    disabled_max = int(np.abs(edited["off"].astype(np.int16) - before).max())
    on = np.rint(before + alpha[..., None] * (edited["on"].astype(np.int16) - edited["off"])).clip(0, 255).astype(np.uint8)
    off = before
    signal = np.abs(chroma(on) - chroma(off)).mean(axis=-1)
    sections = {name: dict(pixels=int(area.sum()), chroma_median=float(np.median(signal[area])),
                           chroma_p95=float(np.percentile(signal[area], 95)), rgb_mae=error(off[area], on[area]))
                for name, area in regions.items()}
    new_endpoints = np.any(((on == 0) & (off > 0)) | ((on == 255) & (off < 255)), axis=-1)
    added_endpoints = float(new_endpoints.mean())
    identical_outside = bool(np.array_equal(off[alpha == 0], on[alpha == 0]))
    admitted = (disabled_max == 0 and identical_outside and added_endpoints <= .02
                and all(sections[name]["pixels"] >= 128 for name in sections)
                and all(sections[name]["chroma_median"] >= 2 for name in ("cheek_left", "cheek_right"))
                and all(sections[name]["rgb_mae"] == 0 for name in ("screen_left", "screen_right")))
    frames = dict(off=off, on=on)
    receipt = dict(qualified=admitted, disabled_max=disabled_max, outside_exact=identical_outside,
                   new_endpoints=added_endpoints, color_policy=policy["policy"], regions=sections,
                   hashes={state: save(folder / f"{state}.png", frame) for state, frame in frames.items()})
    return frames, receipt


def compare(out, subject, donor, photos, regions, engine):
    target = photos[subject]["cold"]["off"]
    wanted = finish_spatial(target, photos[subject]["cold"]["on"], STRENGTH)
    target_path = out / subject / "cold" / "off.png"
    frames = {"stable": {}, "spatial13": {}}
    for light, pair in photos[donor].items():
        frames["stable"][light] = {}
        frames["spatial13"][light] = {}
        for state, reference in pair.items():
            ref_path = out / donor / light / f"{state}.png"
            frames["stable"][light][state] = apply_mapping(target, analyze(target_path, ref_path), STRENGTH)
            frames["spatial13"][light][state] = finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)
    scores = {name: score(target, wanted, results, regions) for name, results in frames.items()}
    for results in scores.values():
        for name in ("screen_left", "screen_right"):
            measure = results[name]
            measure["passed"] = (max(measure["lamp_spill"].values()) <= 3 and all(max(
                row["off_matches_original_mae"], row["on_matches_authored_mae"]) <= 3 for row in measure["response"].values()))
    flat = {name: {f"{light}-{state}": image for light, pair in results.items() for state, image in pair.items()}
            for name, results in ({"reference": photos[donor]} | frames).items()}
    slug = f"{subject}-from-{donor}"
    hashes = {f"{name}-{variant}": save(out / f"{slug}-{name}-{variant}.png", frame)
              for name in frames for variant, frame in flat[name].items()}
    return dict(scores=scores, complete_pass={name: all(row["passed"] for row in result.values())
                                              for name, result in scores.items()}, hashes=hashes,
                wanted=save(out / f"{slug}-expected.png", wanted),
                panel=panel(out / f"{slug}-comparison.png", {"target": {"input": target, "expected": wanted}} | flat))


def run(base, output):
    if any(not path.is_absolute() or not path.as_posix().isascii() for path in (base, output)):
        raise ValueError("Use absolute ASCII paths for darktable and the output")
    archive = base / "gros-remaining-archives-v1"
    capture_parts = checked_json(archive / "verified.json", CAPTURES)["parts"]
    regions_path = Path(__file__).with_name("gros-local-editor-cross-regions.json")
    settings_root = base / "playraw-mixed-light-skin-v1"
    author = settings_root / "author.xmp"
    cli = settings_root / "runtime" / "bin" / "darktable-cli.exe"
    if digest(author) != INPUT_HASHES["author.xmp"] or digest(cli) != INPUT_HASHES["darktable-cli.exe"]:
        raise ValueError("Changed author history or CLI")
    settings = default_states(author.read_bytes())
    region_data = regions_path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(region_data).hexdigest() != REGIONS:
        raise ValueError("Changed region registration")
    coords = json.loads(region_data)
    output.mkdir()
    photos, admission, regions = {}, {}, {}
    for person, status in capture_parts.items():
        root = base / "gros-part-four-recovery-20261122" if person == "4" else archive
        regions[person], alpha = areas_and_weight(coords[person], coords["screen"])
        photos[person], admission[person] = {}, {}
        for light in LIGHTS:
            subject = root / f"part-{person}" / f"{light}.jpg"
            folder = output / person / light
            photos[person][light], admission[person][light] = prepare_frame(
                subject, status["photos"][light]["sha256"], settings, cli, folder, regions[person], alpha)
        print(f"Pair {person}: { {light: row['qualified'] for light, row in admission[person].items()} }", flush=True)
    qualified = [person for person, rows in admission.items() if all(row["qualified"] for row in rows.values())]
    cases, identity = {}, {}
    if len(qualified) >= 2:
        torch.set_num_threads(4)
        engine = SpatialEngine("multiref-v1")
        if engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed model")
        for person in qualified:
            rgb = photos[person]["cold"]["off"]
            path = output / person / "cold" / "off.png"
            identity[person] = dict(stable=bool(np.array_equal(rgb, apply_mapping(rgb, analyze(path, path), STRENGTH))),
                                    spatial13=bool(np.array_equal(rgb, finish_spatial(
                                        rgb, engine.prepare(rgb, rgb).apply(rgb), STRENGTH))))
        for person, donor in permutations(qualified, 2):
            cases[f"{person}-from-{donor}"] = compare(output, person, donor, photos, regions[person], engine)
            print(f"Rendered {person} from {donor}", flush=True)
    json_to(output / "results.json", dict(admission=admission, qualified=qualified, cases=cases, identity=identity,
            captures_sha256=CAPTURES, regions_sha256=REGIONS, strength=STRENGTH,
            author_sha256=INPUT_HASHES["author.xmp"], runtime_sha256=INPUT_HASHES["darktable-cli.exe"],
            stable_fingerprint=algorithm_fingerprint(), checkpoint_sha256=CHECKPOINT,
            protocol_sha256=digest(Path(__file__).with_name("gros-local-editor-cross-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
