"""Find measured small lighting shifts with comparable Oppo capture settings."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import combinations
import json
from math import dist
from pathlib import Path

from audit_beyond_rgb_exposure import PROBE, check_camera, read_tags, select
from audit_beyond_rgb_lab_index import CENTRAL_SHA, MEMBERS
from audit_beyond_rgb_light_pairs import measure


SCENES = ("WHITE_blue", "WHITE_green", "WHITE_red", "WHITE_rich")
CLOSE_LIGHTS = (1, 7, 9)
PROXY = "http://127.0.0.1:7897"
TAG = "WT/oppo_tags.json"


def screen_scene(scene, captures, cache, proxy, cached_only):
    readings = {light: measure(cache, scene, light, captures[scene, light][PROBE], proxy)
                for light in CLOSE_LIGHTS if not cached_only or (cache / f"{scene}-light{light:02d}.range").is_file()}
    tags = {}
    pairs = []
    for a, b in combinations(CLOSE_LIGHTS, 2):
        record = {"lights": [a, b]}
        record["capture_members"] = all(set(MEMBERS) <= captures[scene, light].keys() for light in (a, b))
        pairs.append(record)
        if a not in readings or b not in readings:
            record["pending"] = "instrument_records"
            continue
        first, second = readings[a], readings[b]
        gap = dist(first["xy"], second["xy"])
        ratio = max(first["lux"], second["lux"]) / min(first["lux"], second["lux"])
        checks = {"xy": .008 <= gap <= .035, "lux": ratio <= 1.15}
        record.update(xy_distance=gap, lux_ratio=ratio, checks=checks)
        if all(checks.values()):
            for light in (a, b):
                stem = f"{scene}-{light}-oppo"
                if light not in tags and (not cached_only or (cache / f"{stem}.range").is_file()):
                    tags[light] = read_tags(cache, stem, captures[scene, light][TAG], proxy)
            if a in tags and b in tags:
                record["camera"] = check_camera([tags[a], tags[b]])
            else:
                record["pending"] = "camera_tags"
    return {"measurements": readings, "camera_tags": tags, "pairs": pairs}


def run(central, cache, output, proxy, cached_only):
    captures = select(central, SCENES, MEMBERS, lights=CLOSE_LIGHTS, required_roles=(PROBE, TAG))
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {name: pool.submit(screen_scene, name, captures, cache, proxy, cached_only) for name in SCENES}
        scenes = {name: future.result() for name, future in pending.items()}
    ranked = []
    for name, scene in scenes.items():
        allowed = [row for row in scene["pairs"] if row.get("camera", {}).get("passes") and row["capture_members"]]
        if allowed:
            best = min(allowed, key=lambda row: (abs(row["xy_distance"] - .022), row["lights"]))
            ranked.append((abs(best["xy_distance"] - .022), name, best["lights"]))
    chosen = {name: {light: captures[name, light] for light in lights}
              for _, name, lights in sorted(ranked)[:2]}
    complete = all("pending" not in row for scene in scenes.values() for row in scene["pairs"])
    receipt = {
        "complete": complete,
        "central_sha256": CENTRAL_SHA,
        "protocol_sha256": sha256(Path(__file__).with_name("beyond-rgb-near-white-protocol.md").read_bytes()).hexdigest(),
        "scenes": scenes,
        "selected": chosen if complete else {},
    }
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("central", "cache", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--proxy", default=PROXY)
    parser.add_argument("--cached-only", action="store_true", help="Report verified partial results without a download")
    run(**vars(parser.parse_args()))
