"""Replay aligned reference edits using global or automatically located face effects."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from probe_dast_exposure_response import midtone, save_report, statistics, verified_png
from probe_dast_local_edit_cross import INPUT_SHA, STATES, read_ledger, score_edit, write_png
from segment import SHA256, Segmenter

EDIT_SHA = "5ab0a185c1e4ed6010e196597e5e44875ce9974ba5fa8d7ae742854b07a3aa28"
ARMS = ("global", "face")


def locate(rgb, locator):
    probability, _ = locator.predict(rgb)
    core = cv2.erode((probability[..., 3] >= .70).astype(np.uint8), np.ones((11, 11), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.GaussianBlur(core.astype(float), (25, 25), 4, borderType=cv2.BORDER_CONSTANT).clip(0, 1)


def endpoint_selection(before, delta, weight):
    face = weight >= .9
    full_edit = before[face] + delta[face] / weight[face, None]
    selected = np.all((before[face] >= 2) & (before[face] <= 253)
                      & (full_edit >= 2) & (full_edit <= 253), axis=1)
    accepted = int(selected.sum())
    support = {"face_candidates": int(face.sum()), "face_accepted": accepted, "qualified": accepted >= 1024}
    keep = weight == 0
    keep[face] = selected
    return keep, support


def replay(before, after, donor_weight, target, target_weight, exclude_endpoints=False):
    delta = (after.astype(float) - before).reshape(-1, 3)
    global_shift = delta.mean(axis=0)
    fit = np.column_stack((np.ones(donor_weight.size), donor_weight.ravel()))
    fit_delta = delta
    support = None
    if exclude_endpoints:
        keep, support = endpoint_selection(before.reshape(-1, 3), delta, donor_weight.ravel())
        if support["qualified"]:
            fit, fit_delta = fit[keep], delta[keep]
    coefficients, _, rank, _ = np.linalg.lstsq(fit, fit_delta, rcond=None)
    if rank != 2:
        raise ValueError("Reference has no identifiable face and global components")
    outputs = {}
    for arm, shift in (("global", np.broadcast_to(global_shift, target.shape)),
                       ("face", coefficients[0] + target_weight[..., None] * coefficients[1])):
        change = .8 * shift
        room = np.where(change >= 0, 255. - target, target)
        ratio = np.divide(room, np.abs(change), out=np.ones_like(change), where=change != 0)
        scale = np.minimum(1., ratio.min(axis=2, keepdims=True))
        outputs[arm] = np.round(target + scale * change).astype(np.uint8)
    parameters = {"constant_only": global_shift.tolist(), "constant_with_face": coefficients[0].tolist(),
                  "face": coefficients[1].tolist()}
    if support is not None:
        parameters["support"] = support
    return outputs, parameters


def run(source, prepared, output, exclude_endpoints=False):
    groups = read_ledger(source, "inputs.json", INPUT_SHA)["groups"]
    edits = read_ledger(prepared, "edits.json", EDIT_SHA)["groups"]
    cv2.setNumThreads(1)
    locator = Segmenter()
    output.mkdir()
    rows = []
    try:
        for group in groups:
            donor = str(group["group"])
            target_id = str(group["target_group"])
            target_row = groups[int(target_id) - 1]
            target = verified_png(source / target_id / "nominal.png", target_row["frames"]["nominal"])
            target_weight = locate(target, locator)
            dest = output / donor
            dest.mkdir()
            outputs, hashes, fits = {}, {}, {}
            for state in STATES:
                entry = edits[donor][state]
                before = verified_png(source / donor / f"{state}.png", group["frames"][state])
                after = verified_png(prepared / donor / f"{state}-on.png", entry["edited_sha256"])
                donor_weight = locate(before, locator)
                negative, _ = replay(before, before, donor_weight, target, target_weight, exclude_endpoints)
                for image in negative.values():
                    np.testing.assert_array_equal(image, target)
                outputs[state], fits[state] = replay(before, after, donor_weight, target, target_weight,
                                                     exclude_endpoints)
                for arm, image in outputs[state].items():
                    hashes[f"{state}-{arm}"] = write_png(dest / f"{state}-{arm}.png", image)

            # The frozen evaluation files are opened after all predictions for this group.
            truth_entry = edits[target_id]["nominal"]
            truth = verified_png(prepared / target_id / "truth.png", truth_entry["truth_sha256"])
            outside = verified_png(prepared / target_id / "outside.png", truth_entry["outside_sha256"]) > 0
            core = verified_png(source / target_id / "nominal-mask.png", target_row["target_mask_sha256"]) > 0
            scores, exposure, passes = {}, {}, {}
            for arm in ARMS:
                scores[arm] = {state: score_edit(target, truth, target, outputs[state][arm], core, outside,
                                                edits[donor][state]["source_edit"]) for state in STATES}
                if arm == "face" and exclude_endpoints:
                    for state in STATES:
                        scores[arm][state]["passes"] &= fits[state]["support"]["qualified"]
                nominal = outputs["nominal"][arm]
                exposure[arm] = {}
                for state in STATES[1:]:
                    image = outputs[state][arm]
                    exposure[arm][state] = statistics(nominal, image,
                                                      core & midtone(target) & midtone(nominal) & midtone(image))
                passes[arm] = (all(entry["passes"] for entry in scores[arm].values())
                               and all(entry.get("chroma_norm", np.inf) < .03 for entry in exposure[arm].values()))
            rows.append({"group": int(donor), "fit": fits, "edit": scores, "exposure_on": exposure,
                         "output_sha256": hashes, "passes": passes})
    finally:
        locator.close()
    save_report(output / "results.json", {"inputs_sha256": INPUT_SHA, "edits_sha256": EDIT_SHA,
                "locator_sha256": SHA256, "opencv": cv2.__version__, "negative_pixel_exact": len(rows) * len(STATES),
                "groups_passing": {arm: sum(row["passes"][arm] for row in rows) for arm in ARMS}, "groups": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "prepared", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--exclude-endpoints", action="store_true")
    args = parser.parse_args()
    run(**vars(args))
