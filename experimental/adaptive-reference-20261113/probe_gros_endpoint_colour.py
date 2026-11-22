"""Compare frozen portrait render endpoints using Colour Science 0.4.7."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import colour
from colour.difference import delta_E_CIE2000
import numpy as np
from PIL import Image

REPORTS = {
    "gros-film-b-cross-v2": "4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260",
    "gros-two-image-edit-regression-v1": "6868a57951ec91fd0408fafe1b70a049719e355e4f0ba9a40b026eff01514dd0",
    "local-log-ratio-replay-v1": "4fa419d42d7f12f6a821e737ca964b5d73f5a6e109a7aa94b08997464cda2070",
    "gros-rgb-mixture-v1": "b810f2bcc9ed2a966464c48ae40f63ffa9f4c88564f9faab0ba6f960cf61e5a4",
    "gros-off-budget-v1": "0b08629ad46c0e9deb7d6ca07dc90a4d2f310143ffbf3c3bab4bc554f709f8f4",
}
REGION_SHA = "d69083602c80e65a859dd90531861cd283dffe99de90fb878c981d58315235c5"
LIGHTS = ("cold", "warm")
STATES = ("off", "on")


def checked_bytes(path, fingerprint):
    data = path.read_bytes()
    if sha256(data).hexdigest() != fingerprint:
        raise ValueError(f"Frozen input changed: {path}")
    return data


def frame(path, fingerprint):
    with Image.open(BytesIO(checked_bytes(path, fingerprint))) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (768, 512) or image.info:
            raise ValueError(f"Expected an untagged 768x512 RGB PNG: {path}")
        return np.array(image, dtype=float)


def lab(rgb):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb / 255))


def verify_metric():
    if colour.__version__ != "0.4.7":
        raise ValueError("Reproduce with Colour Science 0.4.7")
    example = delta_E_CIE2000([50, 2.6772, -79.7751], [50, 0, -82.7485])
    np.testing.assert_allclose(example, 2.0425, rtol=0, atol=5e-5)
    gray = lab(np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=float))
    np.testing.assert_allclose(gray[:, 0], [0, 53.585, 100], rtol=0, atol=.002)
    np.testing.assert_array_equal(delta_E_CIE2000(gray, gray), np.zeros(3))
    return float(example)


def compare(light, frames, references, truth_lab, areas, archived_scores):
    endpoints = {
        state: (delta_E_CIE2000(lab(frames[state]), truth_lab[state]),
                np.abs(frames[state] - references[state])) for state in STATES
    }
    rows = {}
    for zone, area in areas.items():
        old = archived_scores[zone]["response"][light]
        endpoint_stats = {}
        for state, previous_key in zip(STATES, ("off_matches_original_mae", "on_matches_authored_mae"), strict=True):
            distances, rgb_error = endpoints[state]
            rgb_mae = float(rgb_error[area].mean())
            np.testing.assert_allclose(rgb_mae, old[previous_key], rtol=0, atol=1e-10,
                                       err_msg=f"{light} {zone} {state}: RGB endpoint differs from archived score")
            median, p95 = np.percentile(distances[area], [50, 95])
            endpoint_stats[state] = dict(delta_e00_median=float(median), delta_e00_p95=float(p95), rgb8_mae=rgb_mae)
        rows[zone] = dict(**endpoint_stats, projection=old["projection"], response_rgb8_mae=old["response_mae"],
                          previous_light_pass=old["passed"], previous_joint_pass=archived_scores[zone]["passed"])
    return rows


def run(root, output):
    control = verify_metric()
    reports = {name: json.loads(checked_bytes(root / name / "results.json", fingerprint))
               for name, fingerprint in REPORTS.items()}
    frozen, continuous, ratios, exact, budget = reports.values()
    regions_path = Path(__file__).with_name("gros-portrait-light-regions.json")
    coords = json.loads(checked_bytes(regions_path, REGION_SHA))
    cases = {}
    for case, origin in frozen["cases"].items():
        target = case.split("-from-", 1)[0]
        baseline_folder = root / "gros-film-b-cross-v2"
        references = {
            "off": frame(baseline_folder / f"{target}-cold-off.png", frozen["inputs"][f"{target}-cold"]["hashes"]["off"]),
            "on": frame(baseline_folder / f"{case}-expected.png", origin["expected_sha"]),
        }
        truth_lab = {key: lab(rgb) for key, rgb in references.items()}
        boxes = coords[target]["cold"]
        areas = {"whole": np.s_[:, :]}
        for zone, (x0, y0, x1, y1) in (coords["screen"] | {z: boxes[z] for z in ("cheek_left", "cheek_right")}).items():
            areas[zone] = np.s_[y0:y1, x0:x1]

        old = continuous["cases"][case]
        ratio = ratios["suites"]["portrait"]["cases"][case]
        exact_case = exact["cases"][case]
        budget_case = budget["cases"][case]
        cases[case] = {}
        for light in LIGHTS:
            old_files = {state: f"two_image-{light}-{role}"
                         for state, role in zip(STATES, ("off", "film_b"), strict=True)}
            files = {state: f"{light}-{state}" for state in STATES}
            arms = (
                ("two_image", root / "gros-two-image-edit-regression-v1", old_files,
                 {s: old["hashes"][name] for s, name in old_files.items()}, ratio["baseline"]),
                ("local_ratios", root / "local-log-ratio-replay-v1" / "portrait", files,
                 {s: ratio["artifacts"][name] for s, name in files.items()}, ratio["scores"]),
                ("exact_off_mix", root / "gros-rgb-mixture-v1", files, exact_case["images"][light], exact_case["scores"]),
                ("budgeted_off_mix", root / "gros-off-budget-v1", files, budget_case["hashes"][light], budget_case["scores"]),
            )
            cases[case][light] = {
                arm: compare(light, {s: frame(folder / f"{case}-{names[s]}.png", hashes[s]) for s in STATES},
                             references, truth_lab, areas, scores)
                for arm, folder, names, hashes, scores in arms
            }

    report = dict(colour_version=colour.__version__, numpy_version=np.__version__, sharma_pair_one=control,
                  reports_sha256=REPORTS, regions_sha256=REGION_SHA, cases=cases)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Existing frozen adaptive-reference results directory")
    parser.add_argument("output", type=Path, help="New output JSON file")
    run(**vars(parser.parse_args()))
