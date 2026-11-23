"""Use the existing direct grid solver on the same frozen local exposure targets."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy
import torch

from probe_gros_context_capacity import PARENT
from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, areas_and_weight
from probe_gros_local_grid_capacity import evaluate, render_case
from probe_gros_mixed_light_exposure_toggle import EV, TRANSFER, apply_exposure
from probe_target_mirror_chroma import CHECKPOINT, checked_json, digest, json_to, panel, save
from spatial_render import SpatialEngine


def run(base, out):
    if scipy.__version__ != "1.18.1":
        raise ValueError("Unexpected SciPy version")
    torch.set_num_threads(4)
    source = base / "gros-mixed-light-exposure-toggle-v1"
    prior = checked_json(source / "results.json", PARENT)
    captures = checked_json(base / "gros-mixed-light-transfer-repeat-v1" / "results.json", TRANSFER)
    code = Path(__file__)
    coordinates = frozen_json(code.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed renderer")
    out.mkdir()
    cases = {}
    for person, receipt in prior["intake"].items():
        input_path = base / "gros-local-editor-channel-v2" / person / "cold" / "off.png"
        original = read_frame(input_path, captures["prepared_hashes"][person]["target"])
        expected = read_frame(source / f"{person}-target-editor-on.png", receipt["target"]["output_sha256"])
        areas, weight = areas_and_weight(coordinates[person], coordinates["screen"])
        areas["outside"] = weight == 0
        raw = apply_exposure(original, weight, EV)
        grades, actual, folds, checks = render_case(engine, original, raw, areas, not cases)
        np.testing.assert_array_equal(actual, expected)
        scores = {key: evaluate(original, expected, values, areas) for key, values in grades.items()}
        complete = {arm: all(item["passed"] for item in rows.values()) for arm, rows in scores.items()}
        cases[person] = dict(scores=scores, complete=complete, folds=folds, controls=checks,
            outputs={key:save(out / f"{person}-{key}.png", values) for key, values in grades.items()},
            board=panel(out / f"{person}-comparison.png", {"direct fit": {"original": original, "expected": expected, **grades}}))
        print(person, complete, flush=True)
    json_to(out / "results.json", dict(source=PARENT, checkpoint=CHECKPOINT, scipy=scipy.__version__,
        script_sha256=digest(code), protocol_sha256=digest(code.with_name("gros-context-grid-control-protocol.md")), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
