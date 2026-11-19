"""Test a single-reference high-frequency cue for physical and digital blue."""
import argparse
from io import BytesIO
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_physical_blue_global_fit import CAPTURE_SHA, DIGITAL_SHA, SCENES, display, linear, pixels, save, verified

ENCODINGS = ("png", "jpeg90", "resize75")
RECEIPT_SHA = "121951adebd16c5b73b68e067bf3523b2794cec582a851664e7243cd584eb04b"


def tile_statistics(values, valid):
    height, width = (n // 16 for n in values.shape[:2])
    rgb = values[:height * 16, :width * 16]
    tiles = rgb.reshape(height, 16, width, 16, 3).transpose(0, 2, 1, 3, 4)
    available = valid[:height * 16, :width * 16].reshape(height, 16, width, 16).all(axis=(1, 3))
    mean = tiles.mean(axis=(2, 3))
    available &= (mean[..., 1:] >= .01).all(axis=-1)
    a, b = tiles[:, :, ::2, ::2], tiles[:, :, ::2, 1::2]
    c, d = tiles[:, :, 1::2, ::2], tiles[:, :, 1::2, 1::2]
    high = (a - b - c + d).reshape(height, width, 64, 3) * .5
    deviation = np.median(np.abs(high - np.median(high, axis=2, keepdims=True)), axis=2)
    low = (a + b + c + d) @ [.2126, .7152, .0722] * .25
    texture = np.square(np.diff(low, axis=2)).mean(axis=(2, 3))
    texture += np.square(np.diff(low, axis=3)).mean(axis=(2, 3))
    return mean, deviation, texture, available


def reading(values, valid):
    mean, mad, texture, available = tile_statistics(values, valid)
    candidates = np.flatnonzero(available)
    count = len(candidates) // 4
    if count < 16:
        return {"value": None, "usable_tiles": len(candidates), "selected_tiles": count}
    weights = texture.ravel()[candidates]
    selected = candidates[np.argsort(weights, kind="stable")[:count]]
    signal = mean.reshape(-1, 3)[selected, 1:]
    noise = mad.reshape(-1, 3)[selected, 1:] ** 2 + 1e-9
    scaled = noise / signal
    value = float(np.median(np.log(scaled[:, 1] / scaled[:, 0])))
    return {"value": value, "usable_tiles": len(candidates), "selected_tiles": count}


def convert(rgb, encoding):
    if encoding == "png":
        return rgb
    image = Image.fromarray(rgb)
    if encoding == "resize75":
        return np.array(image.resize(tuple(round(n * .75) for n in image.size), Image.Resampling.BILINEAR))
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=90, subsampling=2)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return np.array(decoded.convert("RGB"))


def read_rgb(rgb):
    return reading(linear(rgb), ((rgb >= 4) & (rgb <= 251)).all(axis=-1))


def positive_control():
    rng = np.random.default_rng(420)
    shape = (512, 512, 3)
    gain = np.array([1., 1., 2.])
    flux = np.array([3200., 3200., 3200.])
    physical = rng.poisson(flux * gain, size=shape) / 32768.
    digital = rng.poisson(flux, size=shape) * gain / 32768.
    arms = {"physical": physical, "digital": digital}
    return {
        "rgb8": {name: read_rgb(display(rgb)) for name, rgb in arms.items()},
        "linear": {name: reading(rgb, np.ones(rgb.shape[:2], bool)) for name, rgb in arms.items()},
    }


def encoding_control(base, out, capture, captures, edits, measured, folds):
    import cv2

    cv2.setNumThreads(1)
    assets = base / "chromamill-cross-assets-v1"
    receipt = json.loads(verified(assets / "receipt.json", RECEIPT_SHA).read_bytes())
    result = {}
    for scene, level in product(SCENES, (213, 241)):
        other = next(name for name in SCENES if name != scene)
        key = f"{scene}-white-{level}"
        spec = receipt["files"][key]
        encoded = pixels(capture / f"{key}-off.png", captures["prepared"][scene][f"white-{level}"]["hashes"]["off"])
        path = verified(assets / f"{key}.png", spec["sha256"])
        raw = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError("Expected frozen linear RGB8 capture")
        floats = raw[..., ::-1].astype(np.float32) / 255
        height, width = encoded.shape[:2]
        reduced = cv2.resize(floats, (width, height), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(display(reduced), encoded)
        choice = edits["digital"][other][str(level)]
        gains = np.array(choice["gains"])
        digital_before = pixels(base / "cti-selective-correction-v1" / f"{other}-{level}-off-reference.png",
                                choice["states"]["off"]["hashes"]["reference"])
        np.testing.assert_array_equal(display(linear(encoded) * gains), digital_before)
        digital_after = display(reduced * gains)
        mae = float(np.abs(digital_after.astype(np.int16) - digital_before).mean())
        if mae > 1:
            raise ValueError("Encoding-order control changed the image beyond the fixed allowance")
        score = read_rgb(digital_after)
        value = score["value"]
        png = measured[f"{scene}/{level}/off"]
        threshold = folds[scene]["threshold"]
        result[f"{scene}-{level}"] = {
            "before": png["digital"]["png"]["value"], "physical": png["physical"]["png"]["value"],
            "after": value, "mae": mae, "threshold": threshold,
            "lost_classification": threshold is not None and value is not None and value <= threshold,
            "sha256": save(out / f"{scene}-{level}-digital-single-quantization.png", digital_after),
        }
    return result


def run(base, out):
    capture = base / "chromamill-capture-cross-v1"
    edit = base / "cti-selective-correction-v1"
    captures = json.loads(verified(capture / "results.json", CAPTURE_SHA).read_bytes())
    edits = json.loads(verified(edit / "results.json", DIGITAL_SHA).read_bytes())
    control = positive_control()
    control_pass = control["rgb8"]["digital"]["value"] - control["rgb8"]["physical"]["value"] >= .25
    scores = {}
    for scene, level, state in product(SCENES, (213, 241), ("off", "on")):
        target = next(name for name in SCENES if name != scene)
        real = capture / f"{scene}-blue-{level}-{state}.png"
        artificial = edit / f"{target}-{level}-{state}-reference.png"
        choice = edits["digital"][target][str(level)]
        if not choice["qualified"]:
            raise ValueError("Unqualified digital control")
        identity = captures["prepared"][scene][f"blue-{level}"]["hashes"][state]
        variants = {"physical": pixels(real, identity),
                    "digital": pixels(artificial, choice["states"][state]["hashes"]["reference"])}
        scores[f"{scene}/{level}/{state}"] = {
            name: {encoding: read_rgb(convert(rgb, encoding)) for encoding in ENCODINGS}
            for name, rgb in variants.items()
        }
    folds = {}
    for held in SCENES:
        training = [row for key, row in scores.items() if not key.startswith(held + "/") and key.endswith("/off")]
        physical, digital = ([row[name]["png"]["value"] for row in training] for name in ("physical", "digital"))
        threshold = None
        if None not in physical + digital and max(physical) < min(digital):
            threshold = .5 * (max(physical) + min(digital))
        counts = dict.fromkeys(ENCODINGS, 0)
        for key, row in scores.items():
            if not key.startswith(held + "/"):
                continue
            for name, observed in row.items():
                for encoding, measure in observed.items():
                    value = measure["value"]
                    label = None if value is None or threshold is None else "digital" if value > threshold else "physical"
                    measure["prediction"] = label
                    counts[encoding] += int(name == label)
        folds[held] = {"train_physical": physical, "train_digital": digital, "threshold": threshold, "correct": counts}
    if len(scores) != 8:
        raise ValueError("Unexpected reference group count")
    out.mkdir()
    quantization = encoding_control(base, out, capture, captures, edits, scores, folds)
    report = {"parents": [CAPTURE_SHA, DIGITAL_SHA, RECEIPT_SHA], "positive_control": control, "control_pass": control_pass,
              "scores": scores, "folds": folds,
              "quantization_control": quantization,
              "passed": control_pass and all(all(count == 8 for count in fold["correct"].values()) for fold in folds.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
