"""Offline diagnostic for fixed editing support and a fitted global response gain."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_rawtherapee_sky_cross import AREAS, BOXES, GRAD_BOXES, LUMA, STRENGTH, image, score, sources
from probe_tst_neutral_direction_fit import finish_spatial, json_to, load, save
from render_rawtherapee_sky_pair import check

REPORT_SHA = "c4248df0d79d6f5258f792aef72fa20eb8a7ff6216e29a5c6f44907ef95c0299"


def box_slice(shape, box):
    h, w = shape
    x0, y0, x1, y1 = box
    return np.s_[round(h*y0):round(h*y1), round(w*x0):round(w*x1)]


def rgb(path, sha):
    check(path, sha)
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError("Expected RGB archive")
        return np.array(im)


def shift_luma(frame, delta):
    brightened = frame.astype(float) + delta[..., None]
    clipped = ((brightened < 0) | (brightened > 255)).any(axis=-1)
    return np.rint(brightened.clip(0, 255)).astype(np.uint8), float(clipped.mean())


def run(base, out):
    previous = base / "rawtherapee-sky-cross-v2"
    frozen = load(previous / "results.json", REPORT_SHA)
    out.mkdir()
    results = {}
    for name, items in sources(base).items():
        target = image(items[0])
        author = finish_spatial(target, image(items[1]), STRENGTH)
        truth = target, author
        wanted = (author.astype(float) - target) @ LUMA
        support = wanted <= -1
        boxes = dict(zip(AREAS, BOXES["R6_06283"], strict=True)) if name == "R6" else GRAD_BOXES
        sky = box_slice(support.shape, boxes["sky"])
        hashes = frozen["cases"][name]["hashes"]
        frames = {arm: [rgb(previous / f"{name}-{arm}-{s}.png", hashes[f"{arm}-{s}"]) for s in ("off", "on")]
                  for arm in ("13", "regional")}
        rows = {}
        for arm, pair in frames.items():
            old = score(pair, truth, boxes, frames["13"])
            if old != frozen["cases"][name]["scores"][arm]:
                raise ValueError("Archived baseline score changed")
            measured = (pair[1].astype(float) - pair[0]) @ LUMA
            local_delta = np.where(support, measured, 0.)
            x, y = local_delta[sky], wanted[sky]
            energy = np.square(x).sum()
            gain = max(0., float((x*y).sum()/energy)) if energy else 0.
            tested = {}
            for label, scale in (("support", 1.), ("fitted_gain", gain)):
                after, clipped = shift_luma(pair[0], scale * local_delta)
                np.testing.assert_array_equal(after[~support], pair[0][~support])
                tested[label] = score((pair[0], after), truth, boxes, frames["13"]) | {
                    "gain": scale, "clip_fraction": clipped,
                    "sha256": save(out / f"{name}-{arm}-{label}.png", after),
                }
            rows[arm] = {"baseline": old, "controls": tested}
        results[name] = {"support_fraction": float(support.mean()),
                         "sky_support_fraction": float(support[sky].mean()), "arms": rows}
    json_to(out / "results.json", {"parent_sha256": REPORT_SHA, "cases": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
