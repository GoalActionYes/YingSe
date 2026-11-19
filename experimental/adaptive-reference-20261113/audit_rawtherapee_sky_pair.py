"""Verify the RawTherapee state transition and measure preselected image areas."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from render_rawtherapee_sky_pair import SOURCES, check, disabled

LUMA = np.array([.2126, .7152, .0722])
AREAS = ("sky", "sky_side", "foreground_1", "foreground_2", "boundary")
BOXES = {
    "P1005231": ((.035, .192, .145, .235), (.045, .050, .130, .145),
                 (.190, .408, .280, .495), (.720, .730, .900, .810), (.200, .120, .300, .200)),
    "R6_06283": ((.360, .100, .520, .270), (.170, .130, .280, .240),
                 (.700, .920, .830, .970), (.200, .650, .340, .720), (.550, .340, .620, .530)),
}


def read_png(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError("Expected RGB PNG")
        profile = ImageCms.ImageCmsProfile(BytesIO(im.info["icc_profile"]))
        if ImageCms.getProfileDescription(profile).strip() != "RTv4_sRGB":
            raise ValueError("Expected original author sRGB output profile")
        return np.array(im), sha256(im.info["icc_profile"]).hexdigest()


def run(base, out):
    results = {}
    for name, (_, _, profile_sha) in SOURCES.items():
        original = check(base / f"{name}-author.pp3", profile_sha)
        pictures, profiles, fingerprints = [], [], {}
        for state, want in (("off", disabled(original)), ("on", original)):
            child = base / state
            duplicate = base / f"{state}-repeat"
            config = f"{name}-{state}.pp3"
            for directory in (child, duplicate):
                if (directory / config).read_bytes() != want:
                    raise ValueError("Author profile switch mismatch")
            image = f"{name}-{state}.png"
            pixels, profile = read_png(child / image)
            again, again_profile = read_png(duplicate / image)
            if pixels.shape != again.shape or profile != again_profile:
                raise ValueError("Changed repeat geometry or ICC")
            difference = np.abs(pixels.astype(np.int16) - again)
            fingerprints[state] = {
                "png": sha256((child / image).read_bytes()).hexdigest(),
                "pixels": sha256(pixels.tobytes()).hexdigest(),
                "icc": profile,
                "profile": sha256(want).hexdigest(),
                "repeat_changed_channels": int(np.count_nonzero(difference)),
                "repeat_max_channel_difference": int(difference.max()),
            }
            pictures.append(pixels)
            profiles.append(profile)
        if pictures[0].shape != pictures[1].shape or profiles[0] != profiles[1]:
            raise ValueError("Baseline geometry or profile changed")
        before, after = pictures
        height, width = before.shape[:2]
        scores = {}
        for area, (x0, y0, x1, y1) in zip(AREAS, BOXES[name], strict=True):
            box = np.s_[round(height * y0):round(height * y1), round(width * x0):round(width * x1)]
            diff = after[box].astype(float) - before[box]
            delta = diff @ LUMA
            scores[area] = {"pixels": delta.size, "median_luma": float(np.median(delta)),
                            "median_abs_luma": float(np.median(np.abs(delta))),
                            "rgb_mae": float(np.abs(diff).mean())}
        exact_repeat = all(info["repeat_changed_channels"] == 0 for info in fingerprints.values())
        passed = exact_repeat and scores["sky"]["median_luma"] <= -2 and all(
            scores[region]["median_abs_luma"] < 1 for region in ("foreground_1", "foreground_2")
        )
        results[name] = {"shape": [width, height], "fingerprints": fingerprints, "areas": scores,
                         "exact_repeat": exact_repeat, "passed": passed}
    with out.open("x", encoding="utf-8") as stream:
        json.dump({"cases": results, "passed": all(r["passed"] for r in results.values())},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
