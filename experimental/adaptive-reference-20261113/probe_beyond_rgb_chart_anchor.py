"""Check whether measured gray patches preserve the frozen local edit across two lights."""

from argparse import ArgumentParser
from pathlib import Path
from time import monotonic

import numpy as np

from probe_target_mirror_chroma import checked_json, digest, json_to
from run_positional_warmth_trial import ROLES, read_manifest, read_photo


CHARTS = {
    "1": "e0ac8d4406ab3a9fa5a9baa5917f1d6359b9f7394260d30da177a0bce5c05995",
    "7": "5c07ece3246be6272547258e253dc46055db91f58d195a7880a5e3b61219470e",
}
LIGHT = dict(base="7", captured="1", edited="7", captured_edited="1")
PATCHES = (20, 21, 22, 23)


def linear(rgb):
    unit = rgb.astype(np.float64) / 255
    return np.where(unit <= 0.04045, unit / 12.92, ((unit + 0.055) / 1.055) ** 2.4)


def patch_points(corners, shape):
    points = (np.asarray(corners, dtype=np.float64) - 1) / 4
    axis = np.linspace(.25, .75, 21)
    u, v = np.meshgrid(axis, axis)
    weights = np.stack(((1-u)*(1-v), u*(1-v), u*v, (1-u)*v), axis=-1)
    x, y = np.rint(weights @ points).astype(int).reshape(-1, 2).T
    if (x < 0).any() or (y < 0).any() or (x >= shape[1]).any() or (y >= shape[0]).any():
        raise ValueError("Gray patch coordinates lie outside the frozen display")
    return y, x


def gains_from_chart(rgb, chart):
    estimated, records = [], {}
    for number in PATCHES:
        y, x = patch_points(chart[f"patch_{number}"]["corners"], rgb.shape)
        pixels = rgb[y, x]
        valid = ((pixels >= 4) & (pixels <= 251)).all(axis=-1)
        coverage = float(valid.mean())
        row = dict(coverage=coverage, checked_samples=len(pixels),
                   low_counts=(pixels < 4).sum(axis=0).tolist(),
                   high_counts=(pixels > 251).sum(axis=0).tolist())
        records[number] = row
        if coverage < .8:
            continue
        median = np.median(linear(pixels[valid]), axis=0)
        gain = median[1] / median
        estimated.append(gain)
        row.update(median_linear_srgb=median.tolist(), gain=gain.tolist())
    gain = np.median(estimated, axis=0).tolist() if len(estimated) == len(PATCHES) else None
    return dict(gain=gain, patches=records)


def run(base, out):
    start = monotonic()
    family = "beyond_rgb"
    entries = read_manifest()["families"][family]["reference"]
    images = {role: read_photo(base, entries[role], family) for role in ROLES}
    source = base / "beyond-rgb-near-white-v2"
    charts = {light: checked_json(source / f"WHITE_blue-light{light}/oppo_cc_detection.json", sha)
              for light, sha in CHARTS.items()}
    readings = {role: gains_from_chart(rgb, charts[LIGHT[role]]) for role, rgb in images.items()}
    code = Path(__file__)
    if monotonic() - start >= 120:
        raise TimeoutError("The chart eligibility audit exceeded its budget")
    qualified = all(row["gain"] is not None for row in readings.values())
    out.mkdir()
    json_to(out / "results.json", dict(scope="fixed gray chart preflight", charts=CHARTS,
                                       protocol=digest(code.with_name("beyond-rgb-chart-anchor-protocol.md")),
                                       readings=readings, qualified=qualified))
    print(f"Completed chart preflight in {monotonic() - start:.2f} seconds; qualified={qualified}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
