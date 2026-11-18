"""Blindly qualify fixed BG-REAL reserve files, without scoring any model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from itertools import combinations
import json
from pathlib import Path
from shutil import disk_usage

import numpy as np
from PIL import Image, ImageFilter
import requests

from audit_bg_real_reserve import BASE, DEV_SHA, FIT_SHA, FURTHER_LEDGER, INTAKE_SHA, ROOT, pinned, save, used_images
from audit_bg_real_controls import CC_BY, REVISION, verified_csv
from background_intervention import target_regions
from fetch_bg_real_pixel_pilot import LIMIT, REPOSITORY, fetch
from probe_pst50_reserve_fingerprints import distance, fingerprint
from reference_color_app.media import decode_image

RESERVE_SHA = "b6023c29d60066b3a2680eb996a837e93654198855c5cc623b46f4e2f68aaa22"
MAX_BYTES = 24 * 1024**2


def reserve(path):
    value = pinned(path, RESERVE_SHA)
    if value["revision"] != REVISION or len(value["pairs"]) != 12:
        raise ValueError("Unexpected reserve assignment")
    return value["pairs"]


def preflight(job, proxy):
    path = job["path"]
    if not path.startswith(("v2/images/", "v2/masks/subject/")) or ".." in path.split("/"):
        raise ValueError("Unexpected published media path")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    url = f"{REPOSITORY}/resolve/{REVISION}/{path}"
    with requests.head(url, proxies=proxies, allow_redirects=False, timeout=(12, 25)) as response:
        response.raise_for_status()
        first = response.headers
        key = first.get("X-Linked-ETag", first.get("ETag", "")).strip('"')
        size = int(first["X-Linked-Size"] if response.is_redirect else first["Content-Length"])
        if first.get("X-Repo-Commit") != REVISION or len(key) not in (40, 64) or not 0 < size <= LIMIT:
            raise ValueError("Invalid preflight object identity or size")
        if len(key) == 64 and job["expected_sha256"] is not None and key != job["expected_sha256"]:
            raise ValueError("Published payload disagrees with frozen manifest")
    return {**job, "size": size, "release_object": key}


def acquisition(queue, output, proxy):
    attribution, _ = verified_csv(BASE / "bg-real-control-intake", "attribution.csv")
    credits = {row["bgreal_id"]: row for row in attribution}
    jobs, source_credits = [], []
    for i, row in enumerate(queue, 1):
        item = row["reference_items"]
        sources = [credits[key] for key in (row["reference"], item["donor"])]
        if not all(source["License"] == CC_BY for source in sources):
            raise ValueError("Frozen source rights changed")
        source_credits.append(sources)
        for role, arm in item["arms"].items():
            jobs.append({"group": i, "role": role, "path": arm["path"], "expected_sha256": arm["sha256"]})
        jobs.append({"group": i, "role": "mask", "path": item["subject_mask"], "expected_sha256": None})
    with ThreadPoolExecutor(max_workers=6) as threads:
        heads = list(threads.map(lambda row: preflight(row, proxy), jobs))
    total = sum(item["size"] for item in heads)
    if len(heads) != 36 or total > MAX_BYTES or disk_usage(output.parent).free < 2 * total:
        raise ValueError("Reserve media budget failed")
    output.mkdir(exist_ok=True)
    save(output / "preflight.json", heads)

    def download(row):
        name = f'{row["group"]:02d}-{row["role"]}{Path(row["path"]).suffix.lower()}'
        data, record = fetch(row["path"], row["expected_sha256"], proxy, row["size"], row["release_object"])
        path = output / name
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Existing sealed asset differs")
        else:
            with path.open("xb") as handle:
                handle.write(data)
        return {"group": row["group"], "role": row["role"], "file": name, **record}

    with ThreadPoolExecutor(max_workers=6) as threads:
        received = list(threads.map(download, heads))
    save(output / "attribution.json", source_credits)
    save(output / "received.json", {"reserve_sha256": RESERVE_SHA, "revision": REVISION, "size": total,
                                    "files": received})


def read_sources(queue, folder):
    record = json.loads((folder / "received.json").read_bytes())
    if record["reserve_sha256"] != RESERVE_SHA or record["revision"] != REVISION or len(record["files"]) != 36:
        raise ValueError("Unexpected sealed asset receipt")
    if record["size"] != sum(row["size"] for row in record["files"]) or record["size"] > MAX_BYTES:
        raise ValueError("Asset receipt exceeds budget")
    files = {}
    for row in record["files"]:
        group, role = row["group"], row["role"]
        if not 1 <= group <= len(queue) or role not in ("C1", "F1", "mask") or not 0 < row["size"] <= LIMIT:
            raise ValueError("Unexpected reserve asset receipt member")
        manifest = queue[group - 1]["reference_items"]
        path = folder / row["file"]
        key = (group, role)
        if path.parent != folder or path.stat().st_size != row["size"] or key in files:
            raise ValueError("Unexpected reserve asset path or size")
        data = path.read_bytes()
        checked = sha256(data).hexdigest()
        release_key = row["release_object"]
        release_digest = sha1(f"blob {len(data)}\0".encode() + data).hexdigest() if len(release_key) == 40 else checked
        expected_path = manifest["subject_mask"] if role == "mask" else manifest["arms"][role]["path"]
        if (checked != row["sha256"] or release_key != release_digest or row["path"] != expected_path
                or role != "mask" and checked != manifest["arms"][role]["sha256"]):
            raise ValueError("Reserve member differs from the frozen release")
        files[key] = path
    return files, record["size"]


def interior(mask):
    padded = np.pad(mask.astype(np.uint8), 5)
    inner = Image.fromarray(padded).filter(ImageFilter.MinFilter(11))
    return np.asarray(inner)[5:-5, 5:-5] != 0


def check_pixels(queue, files):
    regions, stamps = {}, {}
    for group, row in enumerate(queue, 1):
        width, height = row["reference_items"]["shape"]
        c1, f1 = (decode_image(files[group, role]) for role in ("C1", "F1"))
        with Image.open(files[group, "mask"]) as bitmap:
            mask = np.asarray(bitmap.convert("L")) >= 128
        if c1.shape != f1.shape or c1.shape != (height, width, 3) or mask.shape != (height, width):
            raise ValueError("Release geometry differs")
        core, bg = interior(mask), interior(~mask)
        changes = np.abs(c1.astype(np.int16) - f1)
        changed = {key: float(changes[area].mean()) if area.any() else None
                   for key, area in (("subject", core), ("background", bg))}
        neutral = target_regions(c1, mask)["neutral_proxy"] & core
        people, backdrop, mid = (int(area.sum()) for area in (core, bg, neutral))
        regions[group] = {
            "subject_core_pixels": people, "background_core_pixels": backdrop, "neutral_subject_core_pixels": mid,
            "subject_core_mae": changed["subject"], "background_core_mae": changed["background"],
            "reference_qualified": people >= 256 and backdrop >= 256 and changed["subject"] <= 1
            and changed["background"] >= 8,
            "target_qualified": people >= 256 and mid >= 256,
        }
        for role in ("C1", "F1"):
            stamps[group, role] = fingerprint(files[group, role])
    return regions, stamps


def comparison_sources():
    docs = ROOT / "docs/research"
    record = pinned(docs / "2026-11-18-bg-real-control-intake-evidence.json", INTAKE_SHA)["pairs"]
    used_images(record)
    fitting = BASE / "bg-real-color-directions-inputs"
    dev = BASE / "bg-real-pixel-inputs"
    fit = pinned(fitting / "intake.json", FIT_SHA)
    test = {key: row["input"] for key, row in
            pinned(docs / "2026-11-18-bg-real-pixel-pilot-evidence.json", DEV_SHA)["cases"].items()}
    paths = {"bg-real": [(folder / row[role]["name"], row[role]["sha256"])
                         for folder, ledger in ((fitting, fit), (dev, test))
                         for row in ledger.values() for role in ("C1", "F1")]}
    for name, expected in FURTHER_LEDGER.items():
        directory = BASE / name
        images = pinned(directory / "manifest.json", expected)["images"]
        paths[name] = ([(directory / f'{row["id"]}-input.png', row["input_sha256"]) for row in images]
                       if name == "independent-coco" else
                       [(directory / row["image_file"], row["image_sha256"]) for row in images])
    if {key: len(values) for key, values in paths.items()} != {
        "bg-real": 32, "independent-coco": 60, "supervised-style": 300
    }:
        raise ValueError("Unexpected prior cohorts")
    return paths


def compare(stamps):
    def flags(a, b):
        gap = distance(a["fingerprints"], b["fingerprints"])
        exact = a["bytes_sha256"] == b["bytes_sha256"] or a["pixels_sha256"] == b["pixels_sha256"]
        return gap, exact

    quarantine, links, cross = set(), [], {}
    within_count, within_closest = 0, 64
    for (a, first), (b, second) in combinations(stamps.items(), 2):
        if a[0] == b[0]:
            continue
        gap, exact = flags(first, second)
        within_count += 1
        within_closest = min(within_closest, gap)
        if gap <= 6 or exact:
            quarantine.update((a[0], b[0]))
            links.append({"first": a, "second": b, "distance": gap, "exact": exact})
    for name, files in comparison_sources().items():
        positives, closest = [], 64
        for identity, (path, expected) in enumerate(files, 1):
            old = fingerprint(path)
            if old["bytes_sha256"] != expected:
                raise ValueError("A consumed input differs from its ledger")
            for probe, new in stamps.items():
                gap, exact = flags(new, old)
                closest = min(closest, gap)
                if gap <= 6 or exact:
                    positives.append({"reserve": probe, "prior_ordinal": identity, "gap": gap, "exact": exact})
                    quarantine.add(probe[0])
        cross[name] = {"prior_images": len(files), "comparisons": len(files) * len(stamps),
                       "min_distance": closest, "alerts": positives}
    within = {"comparisons": within_count, "min_distance": within_closest, "alerts": links}
    return quarantine, within, cross


def qualify(queue, inputs, output):
    assets, size = read_sources(queue, inputs)
    rows, stamps = check_pixels(queue, assets)
    quarantine, within, cross = compare(stamps)
    source_position = {row["reference"]: i for i, row in enumerate(queue, 1)}
    valid, excluded = [], {}
    for i, row in enumerate(queue, 1):
        target_index = source_position[row["target"]]
        reasons = [label for label, failed in (
            ("reference_intervention", not rows[i]["reference_qualified"]),
            ("target_coverage", not rows[target_index]["target_qualified"]),
            ("source_overlap", bool({i, target_index} & quarantine)),
        ) if failed]
        if reasons:
            excluded[i] = {"reference_source": i, "target_source": target_index, "reasons": reasons}
        else:
            valid.append(i)
    save(output, {
        "revision": REVISION, "reserve_sha256": RESERVE_SHA,
        "receipt_sha256": sha256((inputs / "received.json").read_bytes()).hexdigest(),
        "attribution_sha256": sha256((inputs / "attribution.json").read_bytes()).hexdigest(),
        "planned_pairs": len(queue), "verified_media_files": len(assets), "stored_media_bytes": size,
        "source_regions": rows, "within_comparison": within, "prior_comparison": cross,
        "isolated_source_ordinals": sorted(quarantine), "technical_pairs": valid, "excluded_pairs": excluded,
        "models_run": 0, "used_for_selection": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "qualify"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--proxy")
    args = parser.parse_args()
    queue = reserve(args.manifest)
    if args.stage == "fetch":
        acquisition(queue, args.assets, args.proxy)
    elif args.output is None:
        parser.error("qualify requires an output JSON path")
    else:
        qualify(queue, args.assets, args.output)
