"""Check the published PSBench central contrast pair and replay the edit if admitted."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from probe_dast_exposure_response import digest, save_report
from probe_inretouch_registration import gradient, shifts
from triplet_color import to_srgb

IDS = ("516", "530")
REC709 = np.array([.2126, .7152, .0722])


def region_masks(rgb):
    h, w = rgb.shape[:2]
    x, y = (np.arange(w)[None, :] + .5) / w, (np.arange(h)[:, None] + .5) / h
    radius = np.maximum(np.abs(x - .5), np.abs(y - .5))
    return radius <= .2, radius <= .25, radius >= .3


def load_pair(folder, task):
    tree = folder / f"{task}-tree.json"
    entries = {Path(row["path"]).name: row for row in json.loads(tree.read_bytes())}
    images, records = [], []
    for side in ("start", "end"):
        name = f"{task}_{side}.jpg"
        path, entry = folder / name, entries[name]
        size, sha = path.stat().st_size, digest(path)
        if size != entry["size"] or sha != entry["lfs"]["oid"]:
            raise ValueError(f"Published file identity mismatch for {name}")
        with Image.open(path) as jpeg:
            rgb, info = to_srgb(ImageOps.exif_transpose(jpeg))
            native = rgb.size
            rgb.thumbnail((512, 512), Image.Resampling.BILINEAR)
            images.append(np.asarray(rgb, dtype=np.float64))
        records.append({"sha256": sha, "bytes": size, "size": native, "color": info["policy"]})
    record = {"task": task, "tree_sha256": digest(tree), "files": records, "eligible": False}
    if records[0]["size"] != records[1]["size"]:
        return images, record
    before, after = images
    order = shifts(gradient(before), gradient(after))[0]
    center, _, outside = region_masks(before)
    delta = np.abs(after - before)
    central_mae, outer_mae = float(delta[center].mean()), float(delta[outside].mean())
    endpoint_fractions = [float(((rgb[center] <= 1) | (rgb[center] >= 254)).any(axis=1).mean())
                          for rgb in images]
    record.update(registration=order, central_author_change=central_mae, outer_author_change=outer_mae,
                  central_endpoint_fraction_start_end=endpoint_fractions,
                  eligible=order["zero"] >= .95 and (order["dy"], order["dx"]) == (0, 0)
                  and central_mae >= 3 and outer_mae <= 1.5 and outer_mae < .35 * central_mae)
    return images, record


def fit(before, after):
    center, _, _ = region_masks(before)
    eligible = center & ((before > 5) & (before < 250) & (after > 5) & (after < 250)).all(axis=2)
    x, y = before[eligible] @ REC709, after[eligible] @ REC709
    if x.size < 1024:
        raise ValueError("Insufficient reference midtones")
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]
    return float(slope), float(intercept)


def render(target, output, slope, intercept):
    center, choice, outside = region_masks(target)
    delta = ((slope - 1.) * (target @ REC709) + intercept)[..., None]
    packed = {}
    for arm, change in (("global", delta), ("central", np.where(choice[..., None], delta, 0.))):
        path = output / f"{arm}.png"
        Image.fromarray(np.rint(np.clip(target + change, 0., 255.)).astype(np.uint8)).save(path)
        packed[arm] = path
    return packed, (center, choice, outside)


def evaluate(before, truth, paths, masks):
    center, _, outside = masks
    baseline = float(np.abs((truth - before)[center]).mean())
    expected = (truth - before)[center] @ REC709
    scores = {}
    for arm, path in paths.items():
        with Image.open(path) as image:
            result = np.asarray(image, dtype=float)
        actual = (result - before)[center] @ REC709
        error = np.abs(result - truth)
        projection = float((expected @ actual) / (expected @ expected))
        central, outer = float(error[center].mean()), float(error[outside].mean())
        scores[arm] = {"central_mae": central, "outer_mae": outer, "projection": projection,
                       "boundary_mae": float(error[~(center | outside)].mean()), "sha256": digest(path),
                       "passes": central <= .75 * baseline and .75 <= projection <= 1.25 and outer <= 1.5}
    return {"baseline_central_mae": baseline, "results": scores}


def run(folder, output):
    output.mkdir()
    loaded = {task: load_pair(folder, task) for task in IDS}
    report = {"source_checks": [record for _, record in loaded.values()], "outputs": {}}
    if all(row["eligible"] for row in report["source_checks"]):
        for target, reference in (IDS, tuple(reversed(IDS))):
            const = fit(*loaded[reference][0])
            before, truth = loaded[target][0]
            dest = output / target
            dest.mkdir()
            paths, masks = render(before, dest, *const)
            report["outputs"][target] = {"reference": reference, "slope": const[0], "intercept": const[1],
                                         **evaluate(before, truth, paths, masks)}
    save_report(output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
