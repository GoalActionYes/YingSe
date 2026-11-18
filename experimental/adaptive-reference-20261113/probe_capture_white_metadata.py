"""Compare DNG capture white metadata with paired raw pixels and JPEG availability."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rawpy

from audit_bg_real_skin_regions import BASE, ROOT, digest


def metadata(path):
    with Image.open(path) as image:
        tags = image.getexif()
        neutral = tags.get(50728)
        mode = tags.get_ifd(34665).get(41987, tags.get(41987))
    values = [float(v) for v in neutral] if isinstance(neutral, (tuple, list)) else None
    valid = values is not None and len(values) == 3 and np.all(np.isfinite(values)) and min(values) > 0
    return {"neutral": values if valid else None, "mode": int(mode) if mode is not None else None}


def pixels(path):
    with rawpy.imread(str(path)) as image:
        array = image.raw_image_visible
        if image.raw_pattern is None or image.raw_pattern.shape != (2, 2):
            raise ValueError("Expected 2x2 Bayer metadata")
        colors = image.color_desc.decode("ascii")
        channels = {label: [] for label in "RGB"}
        for y, x in np.ndindex(2, 2):
            channel = image.raw_colors_visible[y, x]
            black = image.black_level_per_channel[channel]
            sample = array[y:y+32*(array.shape[0]//32):32, x:x+32*(array.shape[1]//32):32].astype(float)
            channels[colors[channel]].append((sample-black) / (image.white_level-black))
        gains = image.camera_whitebalance
        return np.stack([np.mean(channels[label], axis=0) for label in "RGB"], axis=-1), np.array(gains[:3])


def log_chroma(rgb):
    return np.log(rgb[..., [0, 2]] / rgb[..., 1, None])


def compare(ambient, lit):
    a, raw_a = pixels(ambient)
    b, raw_b = pixels(lit)
    ma, mb = metadata(ambient), metadata(lit)
    if ma["neutral"] is None or mb["neutral"] is None:
        raise ValueError("Fixed DNG has no numeric AsShotNeutral")
    for tags, gain in ((ma, raw_a), (mb, raw_b)):
        neutral = np.asarray(tags["neutral"])
        np.testing.assert_allclose((neutral/neutral[1])*(gain/gain[1]), 1, atol=1e-5, rtol=0)
    ambient_ok, lamp_ok = (np.all((image >= .025) & (image <= .85), axis=-1) for image in (a, b))
    usable = ambient_ok & lamp_ok
    whitened = a[usable] / ma["neutral"]
    near_gray = whitened.max(axis=-1) <= 1.15 * whitened.min(axis=-1)
    selected = int(near_gray.sum())
    observed = np.median(log_chroma(b[usable][near_gray])-log_chroma(a[usable][near_gray]), axis=0) if selected else None
    delta = log_chroma(np.asarray(mb["neutral"])) - log_chroma(np.asarray(ma["neutral"]))
    field_strength = float(np.linalg.norm(delta))
    signal = float(np.linalg.norm(observed)) if observed is not None else None
    cosine = float(observed @ delta / (signal*field_strength)) if signal and field_strength else None
    qualifies = selected >= 128 and signal is not None and signal >= .05
    return {"files": {p.name: digest(p) for p in (ambient, lit)}, "ambient": ma, "lamp": mb,
            "observed": None if observed is None else observed.tolist(), "delta": delta.tolist(),
            "sampled": usable.size, "ambient_range": int(ambient_ok.sum()), "lamp_range": int(lamp_ok.sum()),
            "common_range": int(usable.sum()), "paired_points": selected, "signal": signal, "field_strength": field_strength,
            "cosine": cosine, "qualified": qualifies,
            "passed": qualifies and field_strength >= .015 and cosine is not None and cosine >= .8}


def run(out):
    raw = BASE/"synclight-real-pixel-input/raw"
    pairs = {f"{scene}-{view}": compare(raw/f"{scene}_{view}_0_ambient.DNG", raw/f"{scene}_{view}_1_camvis.DNG")
             for scene in ("scene25", "scene35") for view in (1, 2)}
    reference = ROOT/"docs/research/2026-11-18-synclight-real-pixel-evidence.json"
    expected = json.loads(reference.read_bytes())["raw_members"]
    if {name: sha for row in pairs.values() for name, sha in row["files"].items()} != {
        name: value["sha256"] for name, value in expected.items()
    }:
        raise ValueError("Frozen DNG identities changed")
    filters = BASE/"facd-range-media-v1"
    evidence = ROOT/"docs/research/2026-11-18-facd-range-pilot-evidence.json"
    photos = {}
    for key, roles in json.loads(evidence.read_bytes())["cases"].items():
        for style, row in roles.items():
            path = filters/key/(style+".jpg")
            if digest(path) != row["sha256"]:
                raise ValueError("Frozen FACD image changed")
            photos[f"{key}/{style}"] = metadata(path)
    assert len(photos) == 32
    intake = ROOT/"docs/research/2026-11-19-screen-occlusion-intake-evidence.json"
    subset = json.loads(intake.read_bytes())["photos"]
    coco = {}
    for row in subset:
        path = BASE/"screen-occlusion-intake"/f"{row['id']:012}.jpg"
        if digest(path) != row["sha256"]:
            raise ValueError("Frozen COCO image changed")
        coco[row["id"]] = metadata(path)
    assert len(coco) == 9
    result = {"raw_pairs": pairs, "facd": photos, "coco": coco, "passed": all(x["passed"] for x in pairs.values()),
              "source_sha256": digest(Path(__file__)), "protocol_sha256": digest(Path(__file__).with_name(
                  "capture-white-metadata-protocol.md"))}
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
