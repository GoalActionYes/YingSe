"""Check known-look renderability in the product's ordinary RGB CUBE format."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_arri_shared_look import ACES_DISPLAY, ACES_VIEW, CONFIG, LOG, LOOK_SHA, converted, digest, metrics, quantize
from reference_color_app.lut import _sample
from spatial_render import finish_spatial

REPORT_SHA = "34fb0882f660aa8aac7cb7d72fd10e47572d49f78f0a1107f3d6e9f87f6f3b60"
PHOTOS = ("poker_fullshot", "sparks_4592")
SIZE = 64


def transforms(cube):
    import PyOpenColorIO as ocio

    config = ocio.Config.CreateFromBuiltinConfig(CONFIG)
    view = ocio.DisplayViewTransform(src=LOG, display=ACES_DISPLAY, view=ACES_VIEW)
    to_display = config.getProcessor(view).getDefaultCPUProcessor()
    view.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    from_display = config.getProcessor(view).getDefaultCPUProcessor()
    film = config.getProcessor(
        ocio.FileTransform(src=str(cube.resolve()), interpolation=ocio.INTERP_TETRAHEDRAL)
    ).getDefaultCPUProcessor()
    return to_display, from_display, film


def pixel_bound(rgb, desired):
    code = rgb.astype(np.int64) @ [65536, 256, 1]
    _, indices, count = np.unique(code, return_inverse=True, return_counts=True)
    stops = np.cumsum(count)
    middle = stops - (count + 1) // 2
    absolute, inconsistent = 0, np.zeros(len(count), dtype=bool)
    for channel in desired.T.astype(np.int64):
        order = np.lexsort((channel, indices))
        sorted_channel = channel[order]
        median = sorted_channel[middle]
        inconsistent |= sorted_channel[stops - 1] != sorted_channel[stops - count]
        absolute += np.abs(channel - median[indices]).sum()
    return {"mae_lower_bound": float(absolute / desired.size), "distinct_rgb": len(count),
            "ambiguous_rgb": int(inconsistent.sum()), "pixels_at_ambiguous_rgb": int(count[inconsistent].sum())}


def input_photo(folder, name, expected):
    path = folder / name
    if digest(path) != expected:
        raise ValueError(f"Frozen prepared input changed: {name}")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Expected RGB8 image")
        return np.array(image)


def run(folder, official, out):
    parent = folder / "results.json"
    if digest(parent) != REPORT_SHA or digest(official) != LOOK_SHA:
        raise ValueError("Frozen Film A inputs changed")
    baseline = json.loads(parent.read_bytes())
    out.mkdir()
    to_display, from_display, film = transforms(official)

    def look(rgb):
        log = converted(rgb, from_display)
        return np.clip(converted(converted(log, film), to_display), 0, 1)

    space = np.linspace(0, 1, SIZE, dtype=np.float32)
    grid = np.stack(np.meshgrid(space, space, space, indexing="ij"), axis=-1)
    table = look(grid)
    cube = out / "film-a-sdr64.cube"
    header = 'TITLE "Film A SDR display round trip"\nLUT_3D_SIZE 64\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1'
    with cube.open("w", encoding="ascii", newline="\n") as stream:
        np.savetxt(stream, table.transpose(2, 1, 0, 3).reshape(-1, 3), fmt="%.9f", header=header, comments="")

    cases, pooled = {}, {"whole": [], "shared_neutral": []}
    for scene in PHOTOS:
        row = baseline["prepared"][scene]
        original, edited = (input_photo(folder, f"{scene}-{role}.png", row["images"][role])
                            for role in ("off", "on"))
        known = finish_spatial(original, edited, .8)
        raw = original.astype(np.float32) / 255
        direct = finish_spatial(original, quantize(look(raw)), .8)
        baked = finish_spatial(original, quantize(_sample(table, raw)), .8)
        roundtrip = quantize(converted(converted(raw, from_display), to_display))
        errors = np.abs(roundtrip.astype(np.int16) - original)
        neutral = np.logical_and(*(target_regions(pixels, np.zeros(original.shape[:2], dtype=bool))["neutral_proxy"]
                                   for pixels in (original, edited)))
        if neutral.sum() != row["shared_neutral_pixels"]:
            raise ValueError("Frozen neutral region disagrees")
        files = {}
        for role, result in (("direct", direct), ("sdr64", baked)):
            path = out / f"{scene}-{role}.png"
            Image.fromarray(result).save(path)
            files[role] = digest(path)
        cases[scene] = {"hashes": files, "display_roundtrip_mae": float(errors.mean()),
                        "display_roundtrip_max": int(errors.max()), "regions": {}}
        for label, region in (("whole", np.ones(neutral.shape, dtype=bool)), ("shared_neutral", neutral)):
            pooled[label].append((original[region], known[region]))
            old = baseline["predictions"][scene][label]
            measurements = {role: metrics(original, known, original, output, region)
                            for role, output in (("direct", direct), ("sdr64", baked))}
            cases[scene]["regions"][label] = {
                "pixels": int(region.sum()), "same_rgb_bound": pixel_bound(*pooled[label][-1]),
                "old_13": old, **measurements,
                "sdr64_minus_direct_mae": float(np.abs(baked[region].astype(np.int16) - direct[region]).mean()),
            }
    combined = {label: pixel_bound(*(np.concatenate(samples) for samples in zip(*pairs, strict=True)))
                for label, pairs in pooled.items()}
    qualifies = all(row["sdr64"]["passed"] and
                    row["sdr64"]["on_matches_authored_mae"] < row["sdr64"]["original_matches_authored_mae"]
                    for case in cases.values() for row in case["regions"].values())
    results = {"parent_sha256": REPORT_SHA, "look_sha256": LOOK_SHA, "sdr64_sha256": digest(cube),
               "size": SIZE, "offline_known_look_passed": qualifies, "cases": cases, "shared_bound": combined}
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("official_lut", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.prepared, args.official_lut, args.output)
