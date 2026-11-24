"""Measure single-photo vertical luma curvature on matched wall references."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_matched_wall_lightening_workbench import PARENT, SOURCE, load
from probe_playraw_flash_room import LUMA
from probe_shared_authored_tone_admission import ROOM_SHA, support
from probe_target_mirror_chroma import checked_json, digest, json_to


METHODS = ("center", "center_minus_surround")


def patches(center):
    left, top, right, bottom = center
    offset = 2 * (bottom - top)
    return dict(center=center, above=(left, top - offset, right, bottom - offset),
                below=(left, top + offset, right, bottom + offset))


def cue(rgb, areas):
    readings, coverage = {}, {}
    for name, (x0, y0, x1, y1) in areas.items():
        if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
            raise ValueError("Measurement exceeds the frozen image")
        patch = rgb[y0:y1, x0:x1]
        valid = support(patch)
        coverage[name] = float(valid.mean())
        readings[name] = float(np.median(patch[valid] @ LUMA)) if coverage[name] >= .8 else None
    a, b, c = (readings[k] for k in ("center", "above", "below"))
    residual = a - (b + c) / 2 if all(v is not None for v in (a, b, c)) else None
    return dict(center=a, center_minus_surround=residual, medians=readings, coverage=coverage)


def choose(records, method):
    off = [(r[method], tag) for tag, r in records.items() if not r["edited"] and r[method] is not None]
    on = [(r[method], tag) for tag, r in records.items() if r["edited"] and r[method] is not None]
    top_off = max(off, default=(None, None))
    bottom_on = min(on, default=(None, None))
    gap = bottom_on[0] - top_off[0] if on and off else None
    threshold = (bottom_on[0] + top_off[0]) / 2 if gap is not None and gap > 0 else None
    return dict(off_high=top_off, on_low=bottom_on, threshold=threshold, gap=gap,
                eligible_off=len(off), eligible_on=len(on))


def predict(row, method, threshold):
    value = row[method]
    return value is not None and threshold is not None and value > threshold


def leave_room_out(data):
    results = {}
    for held, group in data.items():
        other, = (k for k in data if k != held)
        fold = {}
        for method in METHODS:
            fitted = choose(data[other]["images"], method)
            guesses = {tag: predict(record, method, fitted["threshold"]) for tag, record in group["images"].items()}
            cases = {key: dict(guesses=[guesses[tag] for tag in tags],
                               correct=[guesses[tag] == group["images"][tag]["edited"] for tag in tags])
                     for key, tags in group["directions"].items()}
            fold[method] = dict(train=other, fitting=fitted, guesses=guesses, directions=cases,
                                passed=all(all(row["correct"]) for row in cases.values()))
        results[held] = fold
    return results


def run(base, out):
    source = base / SOURCE
    matched = checked_json(source / "results.json", PARENT)
    room_source = checked_json(base / "mid-gray-anchor-v1.json", ROOM_SHA)["scenes"]
    rooms = {}
    for room, group in matched["rooms"].items():
        directions = {}
        for tag, row in group["comparisons"].items():
            if row["passed"]:
                a, b = tag.split("-to-")
                directions[tag] = (f"{a}-off", f"{b}-off", f"{a}-on")
        tags = sorted({tag for states in directions.values() for tag in states})
        areas = patches(room_source[room]["boxes"]["wall_left"])
        records = {}
        for tag in tags:
            path, rgb = load(source, room, group["renders"], tag)
            records[tag] = dict(cue(rgb, areas), edited=tag.endswith("-on"), sha256=digest(path))
        rooms[room] = dict(directions=directions, areas=areas, images=records)
    if sum(len(room["directions"]) for room in rooms.values()) != 12:
        raise ValueError("Changed cohort")
    folds = leave_room_out(rooms)
    complete = {method: all(fold[method]["passed"] for fold in folds.values()) for method in METHODS}
    out.mkdir()
    code = Path(__file__)
    json_to(out / "results.json", dict(rooms=rooms, folds=folds, complete=complete,
            advance=complete["center_minus_surround"] and not complete["center"],
            parent_sha=PARENT, room_sha=ROOM_SHA, script_sha=digest(code),
            protocol_sha=digest(code.with_name("wall-vertical-luma-protocol.md"))))
    print(complete)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
