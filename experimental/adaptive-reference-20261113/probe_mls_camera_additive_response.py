"""Compare fixed camera-RGB lamp increments with the prior chroma replay."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked, table
from audit_mls_multiscene_intake import SCENES
from fetch_mls_linear_pilot import LIGHTS
from independent_coco import digest
from probe_fau_illuminant_replay import replay
from probe_mls_linear_pilot import read
from probe_mls_material_response import light_gain
from probe_mls_unpaired_material_response import prepare, score


PRIOR_SHA = "db643d7f8dd077ab9d4dfcc83fd23b39b82e1e1491ca4cc96a3daa5bf2df1797"


def camera_centers(region, raw, coded):
    pixels = raw[region]
    exposed = ((coded[region] >= 16) & (coded[region] <= 239)).all(axis=-1)
    return np.median(pixels[exposed].astype(float), axis=0)


def chroma_delta(center, increment):
    adjusted = center + increment
    if not ((adjusted > 0) & (adjusted <= 65535)).all():
        return None
    changes = np.log(adjusted / center)
    return changes[[0, 2]] - changes[1]


def load_scenes(old, added, material):
    with checked(material, "surfaces.zip") as spectra:
        rows = table(spectra, "surfaces/mask_to_surface.csv")
    receipts = {source: json.loads((source / "intake.json").read_bytes())["entries"] for source in (old, added)}
    scenes = {}
    with ZipFile(added / "masks_16bit_color.zip") as masks:
        assert masks.testzip() is None
        names = {Path(entry).name: entry for entry in masks.namelist() if entry.endswith(".png")}
        for scene, material_name in SCENES.items():
            source = old if scene in ("02", "03") else added
            receipt = receipts[source]
            values = prepare(scene, material_name, source, receipt, masks.read(names[f"{scene}.png"]), rows)
            frames, region, *_ = values
            filenames = [f"{scene}_{light}.png" for light in LIGHTS]
            raw_frames = [read(source / filename, receipt[filename]) for filename in filenames]
            centers = [camera_centers(region, raw, coded) for raw, coded in zip(raw_frames, frames, strict=True)]
            scenes[scene] = values, centers
    return scenes


def compare(scene, data, previous, out):
    (frames, region, base, measured, stats), centers = data[scene]
    assert stats == previous["measurement"]
    target = .8 * measured
    donor_offsets = []
    for donors in (previous["same_donors"], previous["other_donors"]):
        donor_offsets.append(np.mean([data[donor][1][1] - data[donor][1][0] for donor in donors], axis=0))
    same, other = donor_offsets
    increments = {"matched": same, "wrong": other, "mixed": (same + other) / 2, "zero": np.zeros(3)}
    arms = {}
    for arm, increment in increments.items():
        offset = chroma_delta(centers[0], increment)
        if offset is None:
            arms[arm] = {"increment": increment.tolist(), "valid": False}
            continue
        rendered, clipped = replay(frames[0], light_gain(offset), region)
        np.testing.assert_array_equal(rendered[~region], frames[0][~region])
        if arm == "zero":
            np.testing.assert_array_equal(rendered, frames[0])
        path = out / f"{scene}-{arm}.png"
        Image.fromarray(rendered).save(path)
        arms[arm] = score(rendered, region, target, base) | {
            "clipped": clipped, "sha256": digest(path), "increment": increment.tolist(),
            "log_chroma": offset.tolist(), "valid": True,
        }
    prior = previous["arms"]["matched"]
    matched = arms["matched"]
    passed = False
    if all(row["valid"] for row in arms.values()):
        errors = matched["quantile_errors"]
        prior_errors = prior["quantile_errors"]
        baseline = arms["mixed"]["quantile_errors"]
        better_controls = all(errors[1] <= .85 * arms[k]["quantile_errors"][1]
                              and errors[1] < arms[k]["quantile_errors"][1] for k in ("wrong", "mixed", "zero"))
        tail_limits = all(errors[i] <= min(baseline[i], prior_errors[i]) + .02 for i in (0, 2))
        threshold = (.85 * prior_errors[1]) if scene == "04" else prior_errors[1] + .015
        coverage = matched["valid_pixels"] >= max(256, stats["interior_pixels"] * .5)
        passed = (better_controls and tail_limits and matched["projection"] is not None
                  and .75 <= matched["projection"] <= 1.25 and errors[1] <= threshold
                  and matched["clipped"] <= prior["clipped"] and coverage)
    return {"measurement_qualified": stats["qualified"], "camera_centers": [x.tolist() for x in centers],
            "old_matched": prior, "arms": arms, "passes": passed}


def run(old, added, material, prior, out):
    assert digest(prior) == PRIOR_SHA
    previous = json.loads(prior.read_bytes())["cases"]
    data = load_scenes(old, added, material)
    out.mkdir()
    cases = {scene: compare(scene, data, previous[scene], out) for scene in SCENES}
    qualified = [scene for scene, row in cases.items() if row["measurement_qualified"]]
    assert qualified == ["02", "03", "04", "09"]
    report = {"source_sha256": digest(Path(__file__)), "prior_sha256": PRIOR_SHA,
              "cases": cases, "passed": all(cases[scene]["passes"] for scene in qualified)}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "added", "material", "prior", "out"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
