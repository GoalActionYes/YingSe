"""Evaluate a fixed target-luma weighting on protected edit increments."""
import argparse
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_beyond_rgb_edit import make_masks
from probe_mid_edit_response_envelope import LEDGER, PARENT, targets
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save


BEYOND = "8ba8333f5bd89afd178523459a1dda02d8029d6fb1d4bd297e289c9baaf91d6c"
LUMA = np.array([.2126, .7152, .0722])


def mid_inputs():
    root = BASE / "mid-matched-wall-cast-v1"
    report = checked_json(root / "results.json", PARENT)
    ledger = checked_json(BASE / "mid-gray-anchor-v1.json", LEDGER)["scenes"]
    for name, (target, regions) in targets(ledger).items():
        case = report["cases"][name]
        expected = checked_rgb(root / f"{name}-expected.png", case["expected_sha"])
        edits = []
        for model in ("stable", "13"):
            hashes = case["models"][model]["hashes"]
            base, edited = [checked_rgb(root / f"{name}-{model}-{state}.png", hashes[state])
                            for state in ("base", "edited")]
            edits.append(edited.astype(np.int16) - base)
        yield name, name, target, expected, regions, edits


def beyond_inputs():
    root = BASE / "beyond-rgb-matched-cast-first-v1"
    prepared = BASE / "beyond-rgb-matched-cast-prepared-v1"
    report = checked_json(root / "report.json", BEYOND)
    ledger = checked_json(prepared / "inputs.json", report["prepared_sha256"])["records"]
    for name, case in report["cases"].items():
        source = ledger[name]
        target = checked_rgb(prepared / f"{name}.png", source["image_sha256"])
        expected = checked_rgb(root / f"{name}-expected.png", case["expected_sha256"])
        edits = []
        for model in ("stable", "spatial13"):
            files = [f"{name}-{model}-{state}.png" for state in ("base", "edit")]
            base, edited = [checked_rgb(root / file, case["outputs_sha256"][file]) for file in files]
            edits.append(edited.astype(np.int16) - base)
        yield name, name.rsplit("-", 1)[0], target, expected, make_masks(target, source["patches"]), edits


def render(target, edits):
    stable, spatial = edits
    weight = (1 - (target @ LUMA) / 255)[..., None]
    increments = {"stable": stable, "13": spatial, "mean": (stable + spatial) / 2,
                  "target_luma": stable + (spatial - stable) * weight}
    return {key: np.rint(np.clip(target + edit, 0, 255)).astype(np.uint8)
            for key, edit in increments.items()}


def run(out):
    out.mkdir()
    cases = {}
    groups = {}
    for cohort in (mid_inputs, beyond_inputs):
        for name, group, target, expected, regions, edits in cohort():
            arms = render(target, edits)
            zero = np.zeros_like(edits[0])
            if not np.array_equal(render(target, (zero, zero))["target_luma"], target):
                raise ValueError("Zero edit should retain identity")
            scores = {zone: {key: metrics(target, expected, target, image, mask)
                             for key, image in arms.items()} for zone, mask in regions.items()}
            flags = {key: all(row[key]["passed"] for row in scores.values()) for key in arms}
            regressions = {}
            for zone, row in scores.items():
                fixed = [score["response_mae"] for key, score in row.items()
                         if key != "target_luma" and score["passed"]]
                outcome = row["target_luma"]
                if fixed and (not outcome["passed"] or outcome["response_mae"] > min(fixed) + 1):
                    regressions[zone] = min(fixed)
            cases[name] = dict(group=group, scores=scores, passed=flags, regressions=regressions,
                               hashes={key: save(out / f"{name}-{key}.png", image)
                                       for key, image in arms.items()},
                               panel=panel(out / f"{name}-comparison.png",
                                           {"target": {"original": target, "desired edit": expected},
                                            "response": arms}))
            groups.setdefault(group, []).append(flags)
    totals = {key: sum(all(row[key] for row in group) for group in groups.values())
              for key in ("stable", "13", "mean", "target_luma")}
    candidate = totals["target_luma"]
    passed = candidate == len(groups) and candidate > max(totals[key] for key in ("stable", "13", "mean"))
    passed = passed and not any(row["regressions"] for row in cases.values())
    source = Path(__file__)
    json_to(out / "results.json", dict(parents={"mid": PARENT, "beyond": BEYOND},
            script_sha=digest(source), protocol_sha=digest(source.with_name("target-luma-edit-mix-protocol.md")),
            group_passes=totals, passed=passed, cases=cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
