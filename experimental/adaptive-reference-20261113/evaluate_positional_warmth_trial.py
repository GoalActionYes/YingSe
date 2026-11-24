"""Fit and evaluate the frozen warm-color trial at native image dimensions."""

from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from evaluate_positional_lightness import PORTRAITS, baseline, evaluate_warmth, save
from probe_arri_shared_look import metrics
from probe_cross_source_wall_lightening_workbench import load
from probe_matched_wall_lightening_workbench import scores
from probe_matched_wall_warmth_workbench import SOURCE, TARGETS, target_areas
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel
from run_positional_warmth_trial import MANIFEST_SHA, ROLES, fit_arm, load_case, read_manifest
from spatial_render import SpatialEngine, finish_spatial
from train_positional_lightness import ConditionedEngine


PROTOCOL = "a64a7406fd073ff8cc51bcf25a6105328f318f1ce84bb92244ecdce04a775d16"
ARMS = ("13", "pooled", "local")
STEPS = 128
AUX_ERRORS = ("off_matches_original_mae", "on_matches_authored_mae", "response_mae",
              "captured_spread_rgb8", "captured_matches_original_mae")


def remaining(deadline):
    if monotonic() >= deadline:
        raise TimeoutError("Exceeded the joint training and export ceiling")


def render_case(render, original, expected, references, out, deadline):
    out.mkdir(parents=True)
    identity = render(original, original)
    outputs = {}
    for role, image in references.items():
        remaining(deadline)
        rendered = render(original, image)
        if rendered.shape != original.shape:
            raise ValueError("The trial requires native-size output")
        outputs[role] = rendered
    hashes = {role: save(out / f"{role}.png", image) for role, image in outputs.items()}
    self_test = dict(exact=bool(np.array_equal(original, identity)), sha256=save(out / "identity.png", identity))
    board = panel(out / "comparison.png", {"target": {"original": original, "expected": expected},
                                           "reference": references, "export": outputs})
    return outputs, dict(outputs=hashes, identity=self_test, board=board)


def difference(first, second, region):
    return float(np.abs(first[region].astype(np.int16) - second[region]).mean())


def heldout_scores(original, expected, outputs, areas):
    result = {}
    for name, region in areas.items():
        state_a = metrics(original, expected, outputs["base"], outputs["edited"], region)
        state_b = metrics(original, expected, outputs["captured"], outputs["captured_edited"], region)
        errors = {f"{state}_{key}": score[key]
                  for state, score in (("a", state_a), ("b", state_b)) for key in AUX_ERRORS[:3]}
        errors.update(off_across_lights=difference(outputs["base"], outputs["captured"], region),
                      on_across_lights=difference(outputs["edited"], outputs["captured_edited"], region))
        passed = max(errors.values()) <= 3 and (name != "wall_left" or (state_a["passed"] and state_b["passed"]))
        result[name] = dict(errors=errors, projection=[state_a["projection"], state_b["projection"]], passed=passed)
    return result


def external_cases(base):
    folder = base / "shared-local-warmth-v2"
    scenes = checked_json(folder / "results.json", SOURCE)["results"]
    for family, group in scenes.items():
        for scene, row in group.items():
            row["folder"] = folder / family / scene

    cases = {}
    for family, name, lamp in TARGETS:
        if family == "mit":
            continue
        scene = scenes[family][name]
        if not scene["frames"][lamp]["metrics"]["admitted"]:
            raise ValueError(f"Unqualified external target {family}/{name}")
        original = load(scene, f"{lamp}-off")[1]
        portrait = family == "gros"
        expected = original if portrait else finish_spatial(original, load(scene, f"{lamp}-on")[1], STRENGTH)
        cases[f"{family}-{name}"] = original, expected, target_areas(original, scene, portrait), portrait

    for name, lamp in PORTRAITS:
        scene = scenes["gros"][name]
        for state in ("off", "on"):
            load(scene, f"{lamp}-{state}")
    return cases, scenes["gros"]


def evaluate(render, case, external, portraits, out, deadline):
    targets, references, spec = case
    original = targets["off"]
    expected = finish_spatial(original, targets["on"], STRENGTH)
    outputs, held = render_case(render, original, expected, references, out / "heldout", deadline)
    measured = heldout_scores(original, expected, outputs, target_areas(original, spec, False))
    held.update(measured=measured, passed=held["identity"]["exact"] and all(row["passed"] for row in measured.values()))

    outside = {}
    active_refs = {name: references[name] for name in ROLES[:3]}
    for key, (original, expected, areas, portrait) in external.items():
        outputs, entry = render_case(render, original, expected, active_refs, out / key, deadline)
        measured = scores(original, expected, outputs, areas)
        if portrait:
            for name, row in measured.items():
                error = difference(outputs["captured"], original, areas[name])
                row.update(captured_matches_original_mae=error, joint_passed=row["joint_passed"] and error <= 3)
        entry.update(measured=measured,
                     passed=entry["identity"]["exact"] and all(row["joint_passed"] for row in measured.values()))
        outside[key] = entry
    remaining(deadline)
    warmth = evaluate_warmth(portraits, render, out / "authored_faces")
    return dict(heldout=held, external=outside, authored_faces=warmth)


def guard_regressions(baseline_report, candidate):
    issues = []
    for name, reference in baseline_report["external"].items():
        current = candidate["external"][name]
        if reference["passed"] and not current["passed"]:
            issues.append(f"{name}/lost")
        for region, prior in reference["measured"].items():
            for key in AUX_ERRORS:
                if key in prior and prior[key] <= 3 < current["measured"][region][key]:
                    issues.append(f"{name}/{region}/{key}")

    for name, reference in baseline_report["authored_faces"].items():
        current = candidate["authored_faces"][name]
        old, new = reference["measurements"]["wall_left"], current["measurements"]["wall_left"]
        if (not current["same_image"]["exact"] or new["projection"] is None or old["projection"] is None
                or abs(new["projection"] - 1) > abs(old["projection"] - 1) or new["response_mae"] > old["response_mae"]):
            issues.append(f"authored-faces/{name}/response")
    for name, row in candidate["external"].items():
        if not row["identity"]["exact"]:
            issues.append(f"{name}/identity")
    return issues


def run(base, out, replay):
    began = monotonic()
    deadline = began + 4 * 3600
    torch.set_num_threads(4)
    script = Path(__file__)
    if digest(script.with_name("positional-warmth-trial-protocol.md")) != PROTOCOL:
        raise ValueError("Changed trial protocol")
    manifest = read_manifest()
    external, portraits = external_cases(base)
    cohorts = {fold["train"]: fold["held_out"] for fold in manifest["folds"]}
    for family in cohorts:
        load_case(base, manifest, family)

    out.mkdir()
    weights = out / "training"
    if replay is None:
        weights.mkdir()
        fits = {}
        for family in cohorts:
            fits[family] = {}
            case = load_case(base, manifest, family)
            for label in ARMS[1:]:
                engine, receipt = fit_arm(case, label, STEPS, deadline)
                path = weights / f"{family}-{label}.pth"
                torch.save(engine.head.state_dict(), path)
                receipt["sha256"] = digest(path)
                fits[family][label] = receipt
                del engine
                print(f"Fitted {family}/{label}", flush=True)
            if fits[family]["pooled"]["sample_order_sha256"] != fits[family]["local"]["sample_order_sha256"]:
                raise ValueError("Training draws differed between the two arms")
        json_to(weights / "fits.json", dict(protocol=PROTOCOL, manifest=MANIFEST_SHA, script=digest(script), folds=fits))
    else:
        weights = replay / "training"

    ledger = json.loads((weights / "fits.json").read_bytes())
    if (ledger["protocol"], ledger["manifest"], ledger["script"]) != (PROTOCOL, MANIFEST_SHA, digest(script)):
        raise ValueError("Training identity mismatch")

    original = SpatialEngine("multiref-v1")
    if original.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13-series checkpoint")
    folds = {}
    for family, heldout in cohorts.items():
        case = load_case(base, manifest, heldout)
        reports = {}
        for label in ARMS:
            remaining(deadline)
            if label == "13":
                render = partial(baseline, engine=original)
            else:
                receipt = ledger["folds"][family][label]
                path = weights / f"{family}-{label}.pth"
                if digest(path) != receipt["sha256"] or receipt["steps"] != STEPS or receipt["checkpoint"] != CHECKPOINT:
                    raise ValueError(f"Changed fitted head {family}/{label}")
                engine = ConditionedEngine(label == "local")
                engine.head.load_state_dict(torch.load(path, map_location=engine.device, weights_only=True), strict=True)
                render = engine.render
            reports[label] = evaluate(render, case, external, portraits, out / family / label, deadline)
            print(f"Evaluated {family}/{label}; complete direction: {reports[label]['heldout']['passed']}", flush=True)
        folds[family] = dict(heldout_family=heldout, arms=reports,
                             regressions=guard_regressions(reports["13"], reports["local"]))

    complete = {label: {family for family, fold in folds.items() if fold["arms"][label]["heldout"]["passed"]}
                for label in ARMS}
    improvements = {label: dict(added=sorted(complete["local"] - complete[label]),
                                lost=sorted(complete[label] - complete["local"])) for label in ARMS[:2]}
    passed = all(row["added"] and not row["lost"] for row in improvements.values())
    remaining(deadline)
    json_to(out / "results.json", dict(protocol=PROTOCOL, checkpoint=CHECKPOINT, training=ledger["folds"],
                                       folds=folds, complete={arm: sorted(cases) for arm, cases in complete.items()},
                                       improvements=improvements,
                                       advances=passed and not any(fold["regressions"] for fold in folds.values())))
    print(f"Total seconds: {monotonic() - began:.2f}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--replay", type=Path, help="Repeat export and scoring with the identified fitted heads")
    run(**vars(parser.parse_args()))
