"""Test a simple spatial lighting cue against mirrored real flash directions."""
from argparse import ArgumentParser
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, quantize
from probe_gros_film_b_cross import FILM
from probe_multilum_gray_probe import EVIDENCE_SHA, LUMA, SCENES, linear, rgb, valid
from probe_target_mirror_chroma import BASE, digest, save


PAIRS = ((0, 2), (1, 3), (4, 5), (8, 21), (9, 20), (10, 22),
         (11, 24), (12, 13), (16, 15), (17, 19))


def measure(frame):
    height, width = frame.shape[:2]
    section = frame[: int(.70 * height)]
    boxes = (section, section[:, :width // 4], section[:, -width // 4:])
    counts, logs = [], []
    for pixels in boxes:
        selected = pixels[valid(pixels)]
        counts.append(len(selected))
        logs.append(float(np.log(np.median(linear(selected) @ LUMA))) if len(selected) >= 1024 else None)
    whole, left, right = logs
    return {"global": whole, "spatial": left - right if left is not None and right is not None else None,
            "valid": counts}


def test_math():
    levels = np.repeat(np.array([128, 64], dtype=np.uint8), 64)
    image = np.broadcast_to(levels[None, :, None], (128, 128, 3))
    a, b = (measure(frame) for frame in (image, image[:, ::-1]))
    assert a["spatial"] > 0 and np.isclose(a["spatial"], -b["spatial"])
    assert a["global"] == b["global"]


def pair_differences(rows, signal, state):
    values = []
    for negative, positive in PAIRS:
        left, right = [rows[str(light)][state][signal] for light in (negative, positive)]
        values.append(right - left if left is not None and right is not None else None)
    return values


def cross_score(source, test):
    labels = {}
    for signal in ("spatial", "global"):
        fitting = [value for value in pair_differences(source, signal, "off") if value is not None]
        sign = float(np.sign(np.median(fitting))) if fitting else 0.
        states = {}
        for mode in ("off", "on"):
            values = pair_differences(test, signal, mode)
            passes = [value is not None and sign * value >= .05 for value in values]
            states[mode] = {"passed": sum(passes), "values": values, "pair_passes": passes}
        labels[signal] = {"calibration_sign": sign, **states}
    return labels


def run(archive_root, evidence, out):
    test_math()
    assert digest(evidence) == EVIDENCE_SHA
    inputs = json.loads(evidence.read_bytes())["archives"]
    cube = BASE / "arri-second-film-assets-v1" / "second-film.cube"
    assert digest(cube) == FILM
    view, inverse, film = transforms(cube)
    out.mkdir()
    results = {}
    for scene in SCENES:
        path = archive_root / f"{scene}.zip"
        assert digest(path) == inputs[scene]
        rows = {}
        with ZipFile(path) as archive:
            metadata = json.loads(archive.read(f"{scene}/meta.json"))
            directions = {row["direction_id"]: row for row in metadata["directions"]}
            for negative, positive in PAIRS:
                a, b = directions[negative], directions[positive]
                assert a["phi"] < 0 < b["phi"]
                assert abs(a["phi"] + b["phi"]) < 1e-4 and abs(a["theta"] - b["theta"]) < 1e-4
            for light in sorted({value for pair in PAIRS for value in pair}):
                frame = rgb(archive.read(f"{scene}/dir_{light}_mip2.jpg"), small=True)
                log = converted(frame.astype(np.float32) / 255, inverse)
                filtered = quantize(np.clip(converted(converted(log, film), view), 0, 1))
                rows[str(light)] = {"off": measure(frame), "on": measure(filtered),
                                   "on_sha256": save(out / f"{scene}-{light:02d}.png", filtered)}
        results[scene] = rows
    validation = {scene: cross_score(results[other], results[scene])
                  for scene, other in (SCENES, SCENES[::-1])}
    passed = all(row["spatial"][state]["passed"] >= 8 and
                 row["spatial"][state]["passed"] - row["global"][state]["passed"] >= 2
                 for row in validation.values() for state in ("off", "on"))
    summary = {"source_sha256": digest(Path(__file__)), "film_sha256": FILM, "pairs": PAIRS,
               "validation": validation, "scenes": results, "passed": passed}
    (out / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("archive_root", "evidence", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
