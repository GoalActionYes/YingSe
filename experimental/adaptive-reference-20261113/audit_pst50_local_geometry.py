"""Blind, input-only geometric near-duplicate check on the 50 PST50 pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
import json
from pathlib import Path

import cv2
import numpy as np

from independent_coco import digest
from probe_pst50_reserve_fingerprints import BASE, REV, ROLES, load_entries, verify
from reference_color_app.media import decode_image

ROLES = ROLES[:2]


def load_image(index, role, inventory):
    row = inventory[role][index]
    if index <= 5:
        folder = BASE / "pst50"
    elif index <= 15:
        folder = BASE / "pst50/development-dense-round2-v1/inputs"
    else:
        folder = BASE / "pst50-reserve-fingerprint-v1/inputs"
    path = folder / row["path"]
    verify(path, row["lfs"]["oid"])
    return decode_image(path)


def descriptors(rgb):
    h, w = rgb.shape[:2]
    k = min(1.0, 640/max(h, w))
    small = cv2.resize(rgb, (round(w*k), round(h*k)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=1200)
    features = []
    for raster in (gray, cv2.flip(gray, 1)):
        points, values = detector.detectAndCompute(raster, None)
        features.append((np.float32([p.pt for p in points]), values, raster.shape))
    return features


def filtered(matcher, source, target):
    return {best.queryIdx: best.trainIdx for best, runner_up in matcher.knnMatch(source, target, k=2)
            if best.distance < .75 * runner_up.distance}


def measure(a, b):
    positions_a, descriptors_a, shape_a = a
    positions_b, descriptors_b, shape_b = b
    row = {"mutual": 0, "inliers": 0, "fit_attempted": False, "passed": False}
    if min(len(positions_a), len(positions_b)) < 20:
        return row
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = filtered(matcher, descriptors_a, descriptors_b)
    if len(forward) < 20:
        return row
    backward = filtered(matcher, descriptors_b, descriptors_a)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    row["mutual"] = len(pairs)
    if len(pairs) < 20:
        return row
    indices_a, indices_b = zip(*pairs, strict=True)
    first, second = positions_a[list(indices_a)], positions_b[list(indices_b)]
    matrix, inliers = cv2.findHomography(first, second, cv2.RANSAC, 3.0)
    row["fit_attempted"] = True
    if matrix is None or only_singular(matrix):
        return row
    selected = inliers.ravel().astype(bool)
    count = int(selected.sum())
    row["inliers"] = count
    if count < 20:
        return row
    points_a, points_b = first[selected], second[selected]
    coverage = [float(cv2.contourArea(cv2.convexHull(points)) / (shape[0]*shape[1]))
                for points, shape in ((points_a, shape_a), (points_b, shape_b))]
    projected = cv2.perspectiveTransform(points_a[None, ...], matrix)[0]
    errors = np.linalg.norm(projected-points_b, axis=1)
    error = float(np.median(errors))
    fraction = count / len(pairs)
    row.update(coverage=coverage, inlier_fraction=fraction, median_error=error,
               passed=fraction >= .55 and min(coverage) >= .03 and error <= 2)
    return row


def only_singular(matrix):
    return not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10


def compare(a, b):
    return {name: measure(a[0], version) for name, version in zip(("normal", "flipped"), b, strict=True)}


def positive_crop(image):
    height, width = image.shape[:2]
    y, x = round(height*.225), round(width*.225)
    cropped = image[y:height-y, x:width-x]
    brighter = np.rint(255*np.sqrt(cropped.astype(np.float32)/255)).astype(np.uint8)
    return np.ascontiguousarray(brighter[:, ::-1])


def run(out):
    cv2.setNumThreads(1)
    inventory = load_entries()
    out.mkdir()
    controls = {}
    cache = {}
    for key in product((1, 3), ROLES):
        image = load_image(*key, inventory)
        cache[key] = descriptors(image)
        synthetic = descriptors(positive_crop(image))
        controls[f"{key[0]}/{key[1]}"] = compare(cache[key], synthetic)
    qualified = all(result["flipped"]["passed"] for result in controls.values())
    result = {"revision": REV, "opencv": cv2.__version__, "controls": controls, "qualified": qualified,
              "protocol_sha256": digest(Path(__file__).with_name("pst50-local-family-protocol.md")),
              "source_sha256": digest(__file__)}
    if qualified:
        keys = list(product(range(1, 51), ROLES))
        for key in keys:
            if key not in cache:
                cache[key] = descriptors(load_image(*key, inventory))
        pairs = [pair for pair in combinations(keys, 2) if pair[0][0] != pair[1][0]]
        if len(pairs) != 4900:
            raise ValueError("Unexpected comparison count")

        def inspect(pair):
            first, second = pair
            stats = compare(cache[first], cache[second])
            return {"first": first, "second": second, "alignment": stats}

        full = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for index, row in enumerate(pool.map(inspect, pairs), 1):
                full.append(row)
                if index % 700 == 0:
                    print(f"Checked {index}/{len(pairs)} image pairs", flush=True)
        (out/"all_pairs.json").write_text(json.dumps(full, allow_nan=False), encoding="utf-8")
        hits = [row for row in full if any(match["passed"] for match in row["alignment"].values())]
        conditional = [row for row in full if row["first"][0] == 1 and row["second"][0] == 3]
        result.update(files=len(keys), comparisons=len(full), hits=hits, source_label_pair_1_3=conditional,
                      fits_attempted=sum(int(match["fit_attempted"]) for row in full for match in row["alignment"].values()),
                      all_pairs_sha256=digest(out/"all_pairs.json"))
    (out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
