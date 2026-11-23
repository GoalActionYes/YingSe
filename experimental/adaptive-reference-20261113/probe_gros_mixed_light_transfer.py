"""Measure fixed-ratio studio light changes through the real image export paths."""
from argparse import ArgumentParser
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from audit_gros_mixed_light_intensity import checked
from probe_arri_shared_look import metrics
from probe_gros_local_editor_channel import actual_png, frozen_json, read_frame, request, serving
from probe_gros_local_editor_cross import REGIONS, SIZE, areas_and_weight
from probe_playraw_mixed_light_transfer import LUMA
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_workbench import make_spatial_workbench
from triplet_color import to_srgb


INTAKE = "266b16091e9ceac3532f24694f5ef5a9d75544736e12c7e5cb245d95820f393f"
LOCAL_EDIT = "0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7"
LEVELS = ("low", "medium", "high")


def difference(first, second, region):
    delta = second[region].astype(np.float64) - first[region]
    brightness = delta @ LUMA
    color = delta - brightness[:, None]
    return dict(rgb8_mae=float(np.abs(delta).mean()), y_prime_mae=float(np.abs(brightness).mean()),
                chroma_mae=float(np.abs(color).mean()), mean_y_prime_shift=float(brightness.mean()),
                mean_rgb8_shift=delta.mean(axis=0).tolist())


def prepare(base, output, intake, frozen):
    images, paths, prepared_hashes = {}, {}, {}
    for subject, group in intake["parts"].items():
        if not group["admission"]["numeric_qualified"]:
            raise ValueError(f"Unqualified capture group: {subject}")
        target = frozen["intake"][subject]["cold"]["hashes"]["off"]
        paths[subject, "target"] = base / "gros-local-editor-channel-v2" / subject / "cold" / "off.png"
        images[subject, "target"] = read_frame(paths[subject, "target"], target)
        if images[subject, "target"].shape != (SIZE[1], SIZE[0], 3):
            raise ValueError("Expected previously fixed target geometry")
        prepared_hashes[subject] = {"target": target}
        for level in LEVELS:
            source = base / "gros-mixed-light-intensity-v1" / f"part-{subject}" / f"{level}.jpg"
            checked(source, group["photos"][level]["sha256"])
            with Image.open(source) as photo:
                rgb, _ = to_srgb(photo)
                pixels = np.array(rgb.resize(SIZE, Image.Resampling.LANCZOS))
            path = output / f"reference-{subject}-{level}.png"
            images[subject, level], paths[subject, level] = pixels, path
            prepared_hashes[subject][level] = save(path, pixels)
    return paths, images, prepared_hashes


def score(target, intended, predictions, cold_off, cold_on, areas):
    rows = {}
    for name, area in areas.items():
        switches = {f"{first}_to_{second}": difference(predictions[first], predictions[second], area)
                    for first, second in combinations(LEVELS, 2)}
        drift = {level: difference(target, predictions[level], area) for level in LEVELS}
        rows[name] = dict(switches=switches, drift_from_original=drift,
                          known_cold_edit=difference(cold_off, cold_on, area),
                          known_cold_edit_response=metrics(target, intended, cold_off, cold_on, area),
                          lamp_changes_pass=all(r["rgb8_mae"] <= 3 for r in switches.values()))
    return rows


def run(base, output):
    torch.set_num_threads(4)
    source = base / "gros-local-editor-channel-v2"
    frozen = checked_json(source / "results.json", LOCAL_EDIT)
    intake = checked_json(base / "gros-mixed-light-intensity-v1" / "results.json", INTAKE)
    if algorithm_fingerprint() != frozen["stable_fingerprint"] or STRENGTH != frozen["strength"]:
        raise ValueError("Stable renderer identity or strength changed")
    coords_path = Path(__file__).with_name("gros-local-editor-cross-regions.json")
    coords = frozen_json(coords_path, REGIONS)
    output.mkdir()
    paths, images, prepared_hashes = prepare(base, output, intake, frozen)
    subjects = tuple(intake["parts"])
    cases = {f"{target}-from-{donor}": {} for target, donor in permutations(subjects)}
    panels = {slug: {} for slug in cases}
    identity = {subject: {} for subject in subjects}
    for model, create in (("stable", make_server), ("spatial13", make_spatial_workbench)):
        parameters = {"enhanced": True} if model == "spatial13" else {}
        with serving(create(output / (model + "-data"), **parameters)) as server:
            if model == "spatial13" and server.store.engine.checkpoint_sha != (frozen["spatial_checkpoint"]):
                raise ValueError("Research checkpoint changed")
            assets = {key: request(server, "/api/media", path.read_bytes(), path.name)["id"]
                      for key, path in paths.items()}
            for subject in subjects:
                key = subject, "target"
                own, fingerprint = actual_png(server, assets[key], assets[key], output / f"{subject}-{model}-same.png")
                np.testing.assert_array_equal(own, images[key])
                identity[subject][model] = fingerprint
            for subject, donor in permutations(subjects):
                slug = f"{subject}-from-{donor}"
                archived = frozen["cases"][slug]
                target = images[subject, "target"]
                expected = read_frame(source / f"{slug}-expected.png", archived["wanted"])
                old = {}
                for state in ("off", "on"):
                    name = f"{model}-cold-{state}"
                    old[state] = read_frame(source / f"{slug}-{name}.png", archived["hashes"][name])
                grades, hashes = {}, {}
                for level in LEVELS:
                    grades[level], hashes[level] = actual_png(server, assets[subject, "target"], assets[donor, level],
                                                              output / f"{slug}-{model}-{level}.png")
                    if grades[level].shape != target.shape:
                        raise ValueError("Workbench changed output geometry")
                areas, _ = areas_and_weight(coords[subject], coords["screen"])
                rows = score(target, expected, grades, old["off"], old["on"], areas)
                cases[slug][model] = dict(outputs=hashes, regions=rows,
                                          passed_lamp_difference=all(row["lamp_changes_pass"] for row in rows.values()))
                panels[slug][model] = grades
                print(f"{slug} {model}: {cases[slug][model]['passed_lamp_difference']}", flush=True)
    boards = {}
    for (subject, donor), (slug, results) in zip(permutations(subjects), panels.items(), strict=True):
        rows = {"target": {"original": images[subject, "target"]},
                "reference": {level: images[donor, level] for level in LEVELS}, **results}
        boards[slug] = panel(output / f"{slug}-comparison.png", rows)
    code = Path(__file__)
    json_to(output / "results.json", dict(source_intake=INTAKE, source_cold_edit=LOCAL_EDIT,
        stable_fingerprint=frozen["stable_fingerprint"], spatial_checkpoint=CHECKPOINT, strength=STRENGTH,
        input_shape=[*SIZE], prepared_hashes=prepared_hashes, cases=cases, identity=identity, boards=boards,
        script_sha256=digest(code), protocol_sha256=digest(code.with_name("gros-mixed-light-transfer-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
