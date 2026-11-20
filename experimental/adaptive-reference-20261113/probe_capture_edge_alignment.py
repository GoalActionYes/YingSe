"""Compare luma/chroma edge alignment under the existing whole-source holdout."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_tile_tone_color import acquire, classify, render
from probe_multilum_gray_probe import chroma, valid
from probe_physical_blue_global_fit import linear
from probe_target_mirror_chroma import LUMA, digest, json_to


def gradients(plane, distance):
    core = slice(distance, -distance)
    horizontal = plane[core, 2 * distance:] - plane[core, :-2 * distance]
    vertical = plane[2 * distance:, core] - plane[:-2 * distance, core]
    return np.stack((horizontal, vertical), axis=-1)


def alignment(luma, chromaticity, support, distance):
    core = slice(distance, -distance)
    eligible = (support[core, core] & support[core, 2 * distance:] & support[core, :-2 * distance] &
                support[2 * distance:, core] & support[:-2 * distance, core])
    dy = gradients(luma, distance)
    dc = gradients(chromaticity, distance)
    a = np.linalg.norm(dy, axis=-1)
    b = np.linalg.norm(dc, axis=-1)
    denominator = a[..., None] * b
    cosines = np.divide(np.abs(np.einsum("...i,...ci->...c", dy, dc)), denominator,
                        out=np.zeros_like(b), where=denominator > 0)
    readings = []
    for channel in range(2):
        color = b[..., channel]
        allowed = eligible & (a >= .005) & (color >= .01)
        if allowed.sum() < 64:
            return None
        for measure in (a, color):
            threshold = np.quantile(measure[allowed], .75)
            mask = allowed & (measure >= threshold)
            readings.append(float(np.median(cosines[..., channel][mask])))
    return readings


def describe(image):
    thumb = Image.fromarray(image)
    thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(thumb)
    support = valid(pixels)
    tone = np.log(np.maximum(linear(pixels) @ LUMA, 1e-6))
    color = chroma(pixels)
    vectors = [alignment(tone, color, support, distance) for distance in (2, 8)]
    return None if any(v is None for v in vectors) else np.r_[vectors[0], vectors[1]]


def run(out):
    sources, mid = acquire()
    features = {family: {name: {state: describe(image) for state, image in frames.items()}
                         for name, frames in group.items()} for family, group in sources.items()}
    decisions = {family: classify(features, family) for family in sources}
    totals = {}
    for family, records in decisions.items():
        states = list(records.values())
        totals[family] = dict(real=sum(row["captured"]["capture"] is True for row in states),
                              digital=sum(row["edited"]["capture"] is True for row in states),
                              base=sum(row["base"]["capture"] is True for row in states),
                              unsupported=sum(item["capture"] is None for row in states for item in row.values()),
                              groups=len(states))
    eligible = all(row["real"] >= .75 * row["groups"] and not (row["digital"] or row["base"] or row["unsupported"])
                   for row in totals.values())
    out.mkdir()
    pictures = render(mid, decisions["mid"]["copyroom"], out)
    src = Path(__file__)
    json_to(out / "results.json", dict(totals=totals, decisions=decisions, pictures=pictures,
            passed=eligible and all(record["passed"] for record in pictures.values()),
            script_sha=digest(src), protocol_sha=digest(src.with_name("capture-edge-alignment-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
