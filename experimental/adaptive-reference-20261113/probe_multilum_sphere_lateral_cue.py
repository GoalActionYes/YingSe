"""Compare analytical sphere geometry and monocular depth on captured gray probes."""
from argparse import ArgumentParser
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, quantize
from probe_gros_film_b_cross import FILM
from probe_multilum_depth_lateral_cue import cue, differences, load_model, save_depth
from probe_multilum_gray_probe import EVIDENCE_SHA, LUMA, SCENES, linear, rgb, valid
from probe_multilum_lateral_cue import PAIRS
from probe_target_mirror_chroma import BASE, digest, save


DIRECT = {2, 3, 19, 20, 21, 22, 24}


def geometry():
    y, x = (np.mgrid[:256, :256] - 127.5) / 127.5
    rr = x * x + y * y
    return x, y, np.sqrt(np.maximum(1 - rr, 0)), rr <= .80**2


def check_geometry():
    x, y, depth, region = geometry()
    left, right = [
        np.log(.15 + np.maximum(.6 * horizontal * x + .25 * y + .75 * depth, 0))
        for horizontal in (-1, 1)
    ]
    s_left, _, _ = cue(depth, left, region)
    s_right, _, _ = cue(depth, right, region)
    assert s_left * s_right < 0 and min(abs(s_left), abs(s_right)) >= .30


def summarize(rows):
    output = {}
    for scene, other in (SCENES, SCENES[::-1]):
        fitting = [d for d in differences(rows[other], "off") if d is not None]
        sign = float(np.sign(np.median(fitting))) if fitting else 0.
        states = {}
        for state in ("off", "on"):
            scores = differences(rows[scene], state)
            pass_list = [score is not None and sign * score >= .05 for score in scores]
            groups = [(a in DIRECT) != (b in DIRECT) for a, b in PAIRS]
            states[state] = {"passed": sum(pass_list), "pair_passes": pass_list, "values": scores,
                             "mixed_visibility": sum(ok and mixed for ok, mixed in zip(pass_list, groups, strict=True)),
                             "same_visibility": sum(ok and not mixed for ok, mixed in zip(pass_list, groups, strict=True))}
        output[scene] = {"calibration_sign": sign, **states}
    return output


@torch.inference_mode()
def run(archive_root, depth_assets, out):
    check_geometry()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = load_model(depth_assets)
    manifest = Path("docs/research/2026-11-18-multilum-pilot-evidence.json")
    assert digest(manifest) == EVIDENCE_SHA
    archives = json.loads(manifest.read_bytes())["archives"]
    cube = BASE / "arri-second-film-assets-v1" / "second-film.cube"
    assert digest(cube) == FILM
    view, inverse, film = transforms(cube)
    _, _, hemisphere, disk = geometry()
    out.mkdir()
    layers = {"exact_sphere": {}, "estimated_depth": {}}
    files = {}
    for scene in SCENES:
        path = archive_root / f"{scene}.zip"
        assert digest(path) == archives[scene]
        arms = {name: {} for name in layers}
        with ZipFile(path) as archive:
            for light in sorted({light for pair in PAIRS for light in pair}):
                key = f"{scene}-{light:02d}"
                photo = rgb(archive.read(f"{scene}/probes/dir_{light}_gray256.jpg"))
                assert photo.shape == (256, 256, 3)
                native = converted(photo.astype(np.float32) / 255, inverse)
                grade = quantize(np.clip(converted(converted(native, film), view), 0, 1))
                files[key] = {"film_sha256": save(out / f"{key}-film.png", grade)}
                for name in arms:
                    arms[name][str(light)] = {}
                for state, frame in (("off", photo), ("on", grade)):
                    estimate = model.infer_image(np.ascontiguousarray(frame[..., ::-1]), input_size=518)
                    files[key][f"{state}_depth_sha256"] = save_depth(out / f"{key}-{state}-depth.png", estimate)
                    sample = disk & valid(frame)
                    y = np.log(np.maximum(linear(frame) @ LUMA, 1e-7))
                    for name, surface in (("exact_sphere", hemisphere), ("estimated_depth", estimate)):
                        value, tiles, count = cue(surface, y, sample)
                        arms[name][str(light)][state] = {"score": value, "tiles": tiles, "pixels": count}
        for name in layers:
            layers[name][scene] = arms[name]
        print(scene, len(files), flush=True)
    scores = {name: summarize(data) for name, data in layers.items()}
    passes = {name: all(row[state]["passed"] >= 8 for row in results.values() for state in ("off", "on"))
              for name, results in scores.items()}
    result = {"scores": scores, "passed": passes, "frames": layers, "hashes": files,
              "source_sha256": digest(Path(__file__)), "film_sha256": FILM,
              "mixed_visibility_pairs": [list(pair) for pair in PAIRS if (pair[0] in DIRECT) != (pair[1] in DIRECT)]}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("archive_root", "depth_assets", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
