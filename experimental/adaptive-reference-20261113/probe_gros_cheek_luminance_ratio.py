"""Test a within-face lightness reference against the authored edit switch."""
from argparse import ArgumentParser
from hashlib import sha256
from math import log
from pathlib import Path

import numpy as np

from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, SIZE, areas_and_weight
from probe_gros_mixed_light_transfer import LEVELS, LOCAL_EDIT
from probe_physical_blue_global_fit import linear
from probe_playraw_flash_room import LUMA, read
from probe_target_mirror_chroma import checked_json, digest, json_to

AUTHOR = "6b66892e5966ab36920c89da11696e1a2067bb43fee13db4b608b5c22c596cd5"
CHEEKS = ("cheek_left", "cheek_right")
ARMS = ("right_luminance", "log_right_left")


def measure(rgb, areas):
    if rgb.dtype != np.uint8 or rgb.shape != (SIZE[1], SIZE[0], 3):
        raise ValueError("Expected frozen RGB8 dimensions")
    values, endpoints = {}, {}
    for name in CHEEKS:
        region = rgb[areas[name]]
        values[name] = float(np.median(linear(region) @ LUMA))
        endpoints[name] = float(((region == 0) | (region == 255)).any(axis=-1).mean())
    left, right = (values[name] for name in CHEEKS)
    if left <= 0 or right <= 0:
        raise ValueError("Nonpositive reference cheek luminance")
    return dict(right_luminance=right, log_right_left=log(right / left),
                left_luminance=left, endpoint_fraction=endpoints)


def measure_person(person, receipt, author_dir, lamp_record, lamp_dir, areas):
    result = {}
    for lamp in ("target", *LEVELS):
        for state in ("off", "on"):
            info = receipt["cases"][lamp]
            path = author_dir / person / lamp / f"editor-{state}.png"
            rgb, _, identity = read(path, False)
            if identity["pixels_sha256"] != info["output_pixels"][state] or identity["icc_sha256"] != info["icc_sha256"]:
                raise ValueError(f"Unexpected author switch: {person} {lamp} {state}")
            result[f"{lamp}-{state}"] = dict(
                measure(rgb, areas), edited=state == "on", rgb_pixels=identity["pixels_sha256"])
    path = lamp_dir / person / "warm/off.png"
    rgb = read_frame(path, lamp_record["warm"]["hashes"]["off"])
    result["warm-off"] = dict(measure(rgb, areas), edited=False, rgb_pixels=sha256(rgb.tobytes()).hexdigest())
    return result


def limits(rows, arm):
    weakest_on = min((row[arm], name) for name, row in rows.items() if row["edited"])
    strongest_off = max((row[arm], name) for name, row in rows.items() if not row["edited"])
    return dict(weakest_on=weakest_on, strongest_off=strongest_off, gap=weakest_on[0] - strongest_off[0])


def evaluate(records, arm):
    split = {}
    for training, testing in (("2", "4"), ("4", "2")):
        support = limits(records[training], arm)
        upper, lower = support["weakest_on"][0], support["strongest_off"][0]
        threshold = (upper + lower) / 2 if upper > lower else None
        tested = {name: dict(truth=row["edited"], predicted=threshold is not None and row[arm] > threshold)
                  for name, row in records[testing].items()}
        split[testing] = dict(training_person=training, support=support, threshold=threshold, decisions=tested,
                              complete=threshold is not None and
                              all(row["truth"] == row["predicted"] for row in tested.values()))
    pooled = {f"{person}/{name}": row for person, conditions in records.items() for name, row in conditions.items()}
    return dict(pooled_threshold_ceiling=limits(pooled, arm), folds=split,
                complete=all(row["complete"] for row in split.values()))


def run(base, out):
    author_dir, lamp_dir = (base / name for name in ("gros-author-tone-switch-v1", "gros-local-editor-channel-v2"))
    author = checked_json(author_dir / "results.json", AUTHOR)
    lamps = checked_json(lamp_dir / "results.json", LOCAL_EDIT)
    source = Path(__file__)
    coordinates = frozen_json(source.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    records = {}
    if not author["admitted"] or set(author["receipts"]) != {"2", "4"}:
        raise ValueError("Changed author switch cohort")
    for person, receipt in author["receipts"].items():
        areas, _ = areas_and_weight(coordinates[person], coordinates["screen"])
        records[person] = measure_person(person, receipt, author_dir, lamps["intake"][person], lamp_dir, areas)
    arms = {arm: evaluate(records, arm) for arm in ARMS}
    out.mkdir()
    json_to(out / "results.json", dict(author_results_sha256=AUTHOR, lamps_results_sha256=LOCAL_EDIT,
            regions_sha256_lf=REGIONS, script_sha256=digest(source),
            protocol_sha256=digest(source.with_name("gros-cheek-luminance-ratio-protocol.md")),
            records=records, arms=arms, advance=arms["log_right_left"]["complete"]))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
