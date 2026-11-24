"""Check whether untouched material can identify lamp leakage from one reference."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_paired_wall_warmth_coordinate import lab
from probe_shared_authored_tone_admission import support
from probe_single_reference_luma_coupling import ORDER, load_inputs, physical_pressure, signed_projection
from probe_target_mirror_chroma import json_to


SCENES = {
    "lion": ((54, 225, 146, 251), (306, 225, 398, 251)),
    "bmug": ((54, 188, 145, 246), (382, 188, 438, 246)),
}


def pixels(photo, box):
    left, top, right, bottom = box
    first_row, first_col = (top + 3) // 4 * 4, (left + 3) // 4 * 4
    return photo[first_row:bottom:4, first_col:right:4]


def floor_pair(photos, weight, boxes):
    readings = []
    for left, top, right, bottom in boxes:
        if np.any(weight[top:bottom, left:right]):
            raise ValueError("Floor reference overlaps the original edit")
        samples = [pixels(img, (left, top, right, bottom)) for img in photos]
        active = np.logical_and.reduce([support(img) for img in samples])
        point_count, coverage = int(active.sum()), float(active.mean())
        unchanged = all(np.array_equal(
            photos[index][top:bottom, left:right], photos[index+2][top:bottom, left:right]
        ) for index in (0, 1))
        row = dict(points=point_count, coverage=coverage, unchanged=unchanged)
        if point_count >= 24 and coverage >= .8 and unchanged:
            row["lab"] = [np.median(lab(img[active]), axis=0).tolist() for img in samples]
        readings.append(row)
    admitted = all("lab" in item for item in readings)
    result = dict(admitted=admitted, floor=readings)
    if admitted:
        contrast = np.subtract(*(item["lab"] for item in readings))[:, 1:]
        result["change"] = (contrast[1] - contrast[0]).tolist()
        result["lightness_change"] = [item["lab"][1][0] - item["lab"][0][0] for item in readings]
    return result


def lamp_names(name):
    _, separator, suffix = name.partition("_")
    return suffix.split("-") if separator else ("w_b", "r_b")


def fit_calibration(rows, scene, feature="change"):
    selected = [(name, row) for name, row in rows.items()
                if name.split("_")[0] != scene and row["floor"]["admitted"] and row["pressure"]["original_vectors"]]
    if len(selected) < 3:
        return dict(eligible=False, pairs=[name for name, _ in selected], reason="insufficient calibration pairs")
    x = np.array([row["floor"][feature] for _, row in selected])
    y = np.array([row["pressure"]["original_vectors"]["physical"] for _, row in selected])
    matrix, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    row = dict(pairs=[name for name, _ in selected], unique_lamp_states=sorted({lamp for name, _ in selected for lamp in lamp_names(name)}),
               rank=int(rank), singular=singular.tolist())
    if rank != 2:
        return dict(**row, eligible=False, reason="floor measurements do not vary in both color dimensions")
    return dict(**row, eligible=True, matrix=matrix.tolist(), calibration_rmse=float(np.sqrt(np.mean((x @ matrix-y)**2))))


def qualify(sources):
    rows = {}
    for name, source in sources.items():
        anonymous, roles, _, weight = source
        scene = name.split("_")[0]
        photos = [anonymous[roles[role]] for role in ORDER]
        floor = floor_pair(photos, weight, SCENES[scene])
        pressure = physical_pressure(source)
        rows[name] = dict(floor=floor, pressure=pressure)
    models = {name: fit_calibration(rows, name) for name in SCENES}
    lightness_models = {name: fit_calibration(rows, name, "lightness_change") for name in SCENES}
    for name, row in rows.items():
        scene = name.split("_")[0]
        calibration = models[scene]
        pressure = row["pressure"]
        if not (row["floor"]["admitted"] and pressure["original_vectors"] and calibration["eligible"]):
            continue
        vector = np.asarray(pressure["original_vectors"]["physical"])
        predicted = np.asarray(row["floor"]["change"]) @ np.asarray(calibration["matrix"])
        remaining = vector - predicted
        old = pressure["original_projection"]
        axes = {key: np.asarray(pressure["original_vectors"][key]) for key in old}
        after = {key: signed_projection(remaining, axis) for key, axis in axes.items()}
        row["held_scene"] = dict(predicted=predicted.tolist(), residual=remaining.tolist(), projection=after,
                                 reduces_both=all(abs(after[k]) < abs(old[k]) for k in axes))
        lightness = lightness_models[scene]
        if lightness["eligible"]:
            secondary = np.asarray(row["floor"]["lightness_change"]) @ np.asarray(lightness["matrix"])
            comparison = {key: signed_projection(vector - secondary, axis) for key, axis in axes.items()}
            row["lightness_only"] = dict(predicted=secondary.tolist(), projection=comparison,
                                         reduces_both=all(abs(comparison[k]) < abs(old[k]) for k in axes))
    return rows, models, lightness_models


def run(base, paired, fau, out):
    _, sources = load_inputs(base, paired, fau, True)
    sources = {name: row for name, row in sources.items() if name.split("_")[0] in SCENES}
    evidence, models, lightness_models = qualify(sources)
    strong = [name for name, row in evidence.items() if row["pressure"]["qualifies"]]
    if len(sources) != 12 or len(strong) != 3:
        raise ValueError("Frozen primary material pressure cohort changed")
    out.mkdir()
    json_to(out / "qualification.json", dict(boxes=SCENES, rows=evidence, models=models,
                                             lightness_models=lightness_models))
    for name in strong:
        row = evidence[name]
        print(name, row["floor"]["admitted"], row.get("held_scene"), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "fau", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
