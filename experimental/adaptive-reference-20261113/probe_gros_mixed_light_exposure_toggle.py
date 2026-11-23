"""Cross a controlled local exposure edit with the three photographed light states."""
from argparse import ArgumentParser
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import torch

from probe_arri_shared_look import metrics
from probe_gros_local_editor_channel import actual_png, frozen_json, read_frame, request, serving
from probe_gros_local_editor_cross import REGIONS, areas_and_weight
from probe_gros_mixed_light_transfer import LEVELS, LOCAL_EDIT, difference
from probe_physical_blue_global_fit import display, linear
from probe_playraw_mixed_light_transfer import LUMA
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


TRANSFER = "035775a2075f7bbcb3e6a65f9414c0b76769abde7f9d7aecc4c8d87f02b0f8db"
EV = .5
CHEEKS = ("cheek_left", "cheek_right")
PROTECTED = ("screen_left", "screen_right", "outside")


def apply_exposure(image, alpha, stops):
    source = linear(image)
    return display(source * (1 + alpha[..., None] * (2 ** stops - 1)))


def admission(original, edited, areas):
    endpoint = np.any(((edited == 0) & (original > 0)) | ((edited == 255) & (original < 255)), axis=2)
    clipped = float(endpoint.mean())
    increase = (edited.astype(np.float64) - original) @ LUMA
    medians = {key: float(np.median(increase[areas[key]])) for key in CHEEKS}
    protected = {key: bool(np.array_equal(edited[areas[key]], original[areas[key]])) for key in PROTECTED}
    return dict(passed=clipped <= .02 and all(m >= 3 for m in medians.values()) and all(protected.values()),
                added_endpoint_fraction=clipped, y_prime_medians=medians, protected_exact=protected)


def measure(target, expected, references, regions):
    results = {}
    changes = {light: on.astype(np.float64) - off for light, (off, on) in references.items()}
    for name, area in regions.items():
        rows = {light: metrics(target, expected, *states, area) for light, states in references.items()}
        spread = {f"{left}_to_{right}": difference(changes[left], changes[right], area)
                  for left, right in combinations(LEVELS, 2)}
        if name in PROTECTED:
            passes = all(max(row["response_mae"], row["off_matches_original_mae"],
                             row["on_matches_authored_mae"]) <= 3 for row in rows.values())
        else:
            passes = all(row["passed"] and row["on_matches_authored_mae"] <= 3 for row in rows.values())
        results[name] = dict(switches=rows, response_differences=spread, passed=passes and
                             all(row["rgb8_mae"] <= 3 for row in spread.values()))
    return results


def run(base, output):
    torch.set_num_threads(4)
    off_root = base / "gros-mixed-light-transfer-repeat-v1"
    cold_root = base / "gros-local-editor-channel-v2"
    frozen = checked_json(off_root / "results.json", TRANSFER)
    cold = checked_json(cold_root / "results.json", LOCAL_EDIT)
    if (algorithm_fingerprint() != frozen["stable_fingerprint"] or
            STRENGTH != frozen["strength"] or CHECKPOINT != frozen["spatial_checkpoint"]):
        raise ValueError("Changed baseline identity")
    code = Path(__file__)
    coords = frozen_json(code.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    output.mkdir()
    image, areas, on_paths, targets, wanted, prepared = {}, {}, {}, {}, {}, {}
    for person, hashes in frozen["prepared_hashes"].items():
        areas[person], alpha = areas_and_weight(coords[person], coords["screen"])
        areas[person]["outside"] = alpha == 0
        target = cold_root / person / "cold" / "off.png"
        if hashes["target"] != cold["intake"][person]["cold"]["hashes"]["off"]:
            raise ValueError("Changed target identity")
        targets[person] = target
        prepared[person] = {}
        for item in ("target", *LEVELS):
            path = target if item == "target" else off_root / f"reference-{person}-{item}.png"
            original = read_frame(path, hashes[item])
            np.testing.assert_array_equal(apply_exposure(original, alpha, 0), original)
            edited = apply_exposure(original, alpha, EV)
            result = admission(original, edited, areas[person])
            image[person, item] = original
            file = output / f"{person}-{item}-editor-on.png"
            if item == "target":
                edited = finish_spatial(original, edited, STRENGTH)
                wanted[person] = edited
            else:
                on_paths[person, item] = file
            prepared[person][item] = dict(**result, output_sha256=save(file, edited))
    admitted = all(v["passed"] for rows in prepared.values() for v in rows.values())
    cases, boards = {}, {}
    if admitted:
        subjects = tuple(targets)
        pairs = list(permutations(subjects, 2))
        cases = {f"{target}-from-{donor}": {} for target, donor in pairs}
        views = {slug: {} for slug in cases}
        for mode, builder, parameters in (("stable", make_server, {}),
                                           ("spatial13", make_spatial_workbench, {"enhanced": True})):
            with serving(builder(output / (mode + "-data"), **parameters)) as server:
                if mode == "spatial13" and server.store.engine.checkpoint_sha != CHECKPOINT:
                    raise ValueError("Changed research checkpoint")
                media = {key: request(server, "/api/media", path.read_bytes(), path.name)["id"]
                         for key, path in on_paths.items()}
                owned = {key: request(server, "/api/media", path.read_bytes(), path.name)["id"]
                         for key, path in targets.items()}
                for target, donor in pairs:
                    slug = f"{target}-from-{donor}"
                    trace, receipts = {}, {}
                    for level in LEVELS:
                        name = f"{slug}-{mode}-{level}"
                        off = read_frame(off_root / f"{name}.png", frozen["cases"][slug][mode]["outputs"][level])
                        on, receipts[level] = actual_png(server, owned[target], media[donor, level], output / f"{name}-on.png")
                        if on.shape != image[target, "target"].shape:
                            raise ValueError("Workbench changed the output size")
                        trace[level] = off, on
                    scores = measure(image[target, "target"], wanted[target], trace, areas[target])
                    cases[slug][mode] = dict(scores=scores, on=receipts, passed=all(row["passed"] for row in scores.values()))
                    views[slug][mode] = {level: trace[level][1] for level in LEVELS}
                    print(slug, mode, cases[slug][mode]["passed"], flush=True)
        for target, donor in pairs:
            slug = f"{target}-from-{donor}"
            rows = {"target": {"original": image[target, "target"], "expected": wanted[target]}, **views[slug]}
            boards[slug] = panel(output / f"{slug}-comparison.png", rows)
    json_to(output / "results.json", dict(source_transfer=TRANSFER, cold_source=LOCAL_EDIT, ev=EV,
        strength=STRENGTH, checkpoint=CHECKPOINT, stable_fingerprint=frozen["stable_fingerprint"],
        admitted=admitted, intake=prepared, cases=cases, boards=boards, script_sha256=digest(code),
        protocol_sha256=digest(code.with_name("gros-mixed-light-exposure-toggle-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
