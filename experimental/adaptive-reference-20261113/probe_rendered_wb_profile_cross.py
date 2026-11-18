"""Test additive profile reuse across published white-balance settings."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_inretouch_registration import gradient, shifts
from probe_reference_chroma_spread import projection
from probe_bg_real_editor_look import save_panel

COMMIT = "d70afa227c94152fd8421db75b3ee6ac06a417dc"
REPO = "mahmoudnafifi/Deep_White_Balance"
WBS = ("C", "D", "F", "S", "T")
STYLES = ("AS", "CS")
BUDGET = 6 * 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_inputs(folder, proxy):
    opener = build_opener(ProxyHandler({"https": proxy} if proxy else {}))

    def get(url, max_bytes):
        with opener.open(url, timeout=30) as response:
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Request exceeds declared size: {url}")
        return data

    api = f"https://api.github.com/repos/{REPO}/git"
    commit = json.loads(get(f"{api}/commits/{COMMIT}", 32_768))
    tree_id = commit["tree"]["sha"]
    tree = json.loads(get(f"{api}/trees/{tree_id}?recursive=1", 100_000))
    if commit["sha"] != COMMIT or tree["sha"] != tree_id or tree["truncated"]:
        raise ValueError("Unexpected source revision")
    by_path = {row["path"]: row for row in tree["tree"]}
    paths = [f"dataset/8D5U5524_{wb}_{style}.png" for wb in WBS for style in STYLES]
    if sum(by_path[name]["size"] for name in paths) >= BUDGET:
        raise ValueError("Image budget exceeded")
    base = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
    folder.mkdir()
    entries = {}
    for name in paths:
        blob = by_path[name]
        data = get(base + name, blob["size"])
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if (len(data), git_blob) != (blob["size"], blob["sha"]):
            raise ValueError(f"Source identity mismatch: {name}")
        file = Path(name).name
        (folder/file).write_bytes(data)
        entries[file] = {"bytes": len(data), "blob": git_blob, "sha256": sha256(data)}
    manifest = {"commit": COMMIT, "tree": tree_id, "files": entries}
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def run(source, out):
    manifest = json.loads((source/"manifest.json").read_bytes())
    if manifest["commit"] != COMMIT or len(manifest["files"]) != 10:
        raise ValueError("Expected frozen source")
    photos, sizes, profiles, orientations = {}, set(), set(), set()
    for wb in WBS:
        for style in STYLES:
            name = f"8D5U5524_{wb}_{style}.png"
            path = source / name
            if sha256(path.read_bytes()) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Source image changed: {name}")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"Expected RGB8: {name}")
                sizes.add(image.size)
                profiles.add(image.info.get("icc_profile"))
                orientations.add(image.getexif().get(274, 1))
                photos[wb, style] = np.asarray(image).copy()
    if len(sizes) != 1 or len(profiles) != 1 or orientations != {1}:
        raise ValueError("Source encoding or raster layout differs")
    gradients = {}
    for key, pixels in photos.items():
        thumb = Image.fromarray(pixels)
        thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
        gradients[key] = gradient(np.asarray(thumb))
    anchor = gradients["D", "AS"]
    alignment = {f"{wb}_{style}": shifts(anchor, item)[0]
                 for (wb, style), item in gradients.items() if (wb, style) != ("D", "AS")}
    registered = all(r["zero"] >= .5 and (r["dx"], r["dy"]) == (0, 0) for r in alignment.values())
    out.mkdir()
    results = {}
    fixed = photos["D", "CS"].astype(np.int16) - photos["D", "AS"]
    if registered:
        for wb in ("C", "F", "S", "T"):
            source_as, expected = photos[wb, "AS"], photos[wb, "CS"]
            before = source_as.astype(np.int16)
            target = expected.astype(np.int16)
            raw = before + fixed
            adjusted = np.clip(raw, 0, 255).astype(np.uint8)
            person = np.zeros(before.shape[:2], bool)
            source_regions, target_regions_ = target_regions(source_as, person), target_regions(expected, person)
            regions = {"whole": source_regions["whole"],
                       "common_neutral": source_regions["neutral_proxy"] & target_regions_["neutral_proxy"]}
            result = {}
            for name, region in regions.items():
                if region.sum() < 256:
                    result[name] = None
                    continue
                base = float(np.abs(before[region]-target[region]).mean())
                error = float(np.abs(adjusted[region]-target[region]).mean())
                result[name] = dict(pixels=int(region.sum()), original_mae=base, predicted_mae=error,
                                    residual_projection=projection(fixed, target-before, region),
                                    clipped=float(((raw[region]<0) | (raw[region]>255)).mean()))
            pair = out/f"{wb}-comparison.png"
            predicted_file = out/f"{wb}-reuse.png"
            Image.fromarray(adjusted).save(predicted_file)
            save_panel(pair, {"original": {"AS": source_as, "published CS": expected},
                              "D residual reused": {"estimate": adjusted, "published CS": expected}})
            results[wb] = {"regions": result, "passed": all(r is not None and r["predicted_mae"] <= 1.5
                           and r["predicted_mae"] < r["original_mae"] for r in result.values()),
                           "png_sha256": sha256(predicted_file.read_bytes()),
                           "panel_sha256": sha256(pair.read_bytes())}
    report = {"input": manifest, "raster": list(sizes.pop()),
              "icc_sha256": sha256(profile) if (profile := profiles.pop()) is not None else None,
              "registration": alignment, "registered": registered, "cases": results,
              "passed": len(results) == 4 and all(r["passed"] for r in results.values()),
              "source_sha256": sha256(Path(__file__).read_bytes()),
              "protocol_sha256": sha256(Path(__file__).with_name("rendered-wb-profile-cross-protocol.md").read_bytes())}
    (out/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--proxy")
    args = parser.parse_args()
    if args.fetch:
        fetch_inputs(args.input, args.proxy)
    run(args.input, args.output)
