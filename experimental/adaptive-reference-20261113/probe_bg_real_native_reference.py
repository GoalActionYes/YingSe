"""Isolate the reference resampling step on the frozen HSL/curve experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_bg_real_editor_look import BASE, PARENT, PARENT_SHA, ROOT, STATES, STRENGTH, digest, evaluate, mask_for, save_panel
from reference_color_app.media import decode_image
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial

PARENT_RESULT_SHA = "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d"
ROLES = ("C1", "F1")


def verified_image(folder, filename, evidence):
    path = folder / filename
    if digest(path) != evidence[filename]:
        raise ValueError(f"Frozen input or baseline changed: {path}")
    return decode_image(path)


def render_case(engine, target, source, baseline):
    histogram = engine.api["batch_histogram_1D"]
    convert = engine.api["srgb_to_ycbcr"]
    tensor = engine.analysis(target)
    target_hist = histogram(convert(tensor))
    low = engine.api["TF"].resize(tensor, (64, 64), antialias=True)
    output = {"control": {}, "native": {}}
    hist_deltas = {"control": {}, "native": {}}
    sizes = {}
    for role in ROLES:
        output["control"][role], output["native"][role] = {}, {}
        hists = {"control": {}, "native": {}}
        for state in STATES:
            image = source[f"{role}-{state}-reference.png"]
            sizes[role] = [image.shape[1], image.shape[0]]
            native = engine.tensor(image)
            hists["control"][state] = histogram(convert(engine.analysis(image)))
            hists["native"][state] = histogram(convert(native))
            for arm in output:
                condition = torch.cat((target_hist, hists[arm][state]), dim=1)
                grid = engine.api["lowpass"](engine.model.apply_grid(low, condition))
                pixels = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)
                if arm == "control":
                    np.testing.assert_array_equal(pixels, baseline[f"{role}-{state}-13.png"])
                output[arm][role][state] = pixels
        for arm, values in hists.items():
            hist_deltas[arm][role] = (.5 * (values["on"] - values["off"]).abs().sum(dim=-1))[0].tolist()
    return output, hist_deltas, sizes


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    folder = BASE / "bg-real-editor-look"
    latest = ROOT / "docs/research/2026-11-19-bg-real-editor-look-evidence.json"
    parent = PARENT / "results.json"
    if (digest(latest), digest(parent)) != (PARENT_RESULT_SHA, PARENT_SHA):
        raise ValueError("Development evidence changed")
    evidence = json.loads(latest.read_bytes())
    targets = {row["pair"]: row for row in json.loads(parent.read_bytes())["cases"]}
    engine = SpatialEngine("multiref-v1")
    if (engine.checkpoint_sha, STRENGTH) != (evidence["model_sha256"], evidence["strength"]):
        raise ValueError("Frozen model or strength changed")
    out.mkdir()
    records = {}
    improvement, regressions = [], []
    for pair, case in evidence["cases"].items():
        artifacts = case["artifacts"]
        inputs = {name: verified_image(folder/pair, name, artifacts)
                  for name in artifacts if name.endswith("-reference.png")}
        baseline = {name: verified_image(folder/pair, name, artifacts)
                    for name in artifacts if name.endswith("-13.png")}
        target = verified_image(folder/pair, "target.png", artifacts)
        images, hist_deltas, sizes = render_case(engine, target, inputs, baseline)
        truth = {"off": target, "on": verified_image(folder/pair, "expected-on.png", artifacts)}
        mask = mask_for(targets[pair], pair.split("-")[0])
        metrics = {arm: evaluate(target, truth, photos, mask) for arm, photos in images.items()}
        control, native = metrics["control"], metrics["native"]
        for role in ROLES:
            difference = abs(1 - control["projection"][role]["luma"]) - abs(1 - native["projection"][role]["luma"])
            improvement.append(difference)
        for state in STATES:
            for region, before in control["swap_mae"][state].items():
                if before is not None:
                    regressions.append(native["swap_mae"][state][region] - before)
        case_dir = out / pair
        case_dir.mkdir()
        rows = {"expected": truth}
        for arm, photos in images.items():
            for role, states in photos.items():
                rows[f"{arm} {role}"] = states
                if arm == "native":
                    for state, pixels in states.items():
                        Image.fromarray(pixels).save(case_dir / f"{role}-{state}-native.png")
        save_panel(case_dir/"comparison.png", rows)
        records[pair] = {"reference_sizes": sizes, "histogram_on_off_total_variation": hist_deltas,
                         "metrics": metrics,
                         "hashes": {p.name: digest(p) for p in sorted(case_dir.glob("*.png"))}}

    improved_count = sum(delta >= .2 for delta in improvement)
    worse = max(regressions)
    passing = sum(row["metrics"]["native"]["passed"] for row in records.values())
    maximum = max(row["metrics"]["native"]["max_protected_swap"] for row in records.values())
    report = {"cases": records, "luma_improved_at_least_point_two": improved_count,
              "max_protected_regression": worse, "native_passing_cases": passing,
              "native_max_protected_swap": maximum,
              "promising_resize_loss": improved_count >= 6 and worse <= 1,
              "passed": passing >= 3 and maximum <= 3, "model_sha256": engine.checkpoint_sha,
              "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("bg-real-native-reference-protocol.md"))}
    (out/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
