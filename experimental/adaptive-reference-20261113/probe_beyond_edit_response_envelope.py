"""Bound the edit attainable by mixing two saved static-scene responses."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_mid_edit_response_envelope import clipped, solution
from probe_target_luma_edit_mix import BEYOND, beyond_inputs
from probe_target_mirror_chroma import digest, json_to, panel, save


def projection_bounds(start, end, target, expected):
    signal = expected - target
    norm = float(np.square(signal).sum())
    if not norm:
        raise ValueError("No authored change in scoring region")
    step = end - start

    def dot(rgb):
        return ((clipped(rgb) - target) * signal).sum(axis=-1)

    low = np.minimum(dot(start), dot(end))
    high = np.maximum(dot(start), dot(end))
    # Clipping makes projection piecewise linear in the shared RGB weight.
    for limit in (0, 255):
        roots = np.divide(limit - start, step, out=np.zeros_like(step), where=step != 0)
        for weight in np.clip(roots, 0, 1).T:
            measure = dot(start + weight[:, None] * step)
            np.minimum(low, measure, out=low)
            np.maximum(high, measure, out=high)
    margin = .5 * float(np.abs(signal).sum())
    return [(float(low.sum()) - margin) / norm, (float(high.sum()) + margin) / norm]


def run(out):
    out.mkdir()
    cases = {}
    for name, group, target, expected, regions, edits in beyond_inputs():
        origin = target.astype(np.float64)
        desired = expected.astype(np.float64)
        left, right = (origin + edit for edit in edits)
        continuous, _ = solution(left, right, desired)
        best = np.rint(continuous).astype(np.uint8)
        stable, spatial = (clipped(rgb).astype(np.uint8) for rgb in (left, right))
        arms = {"stable": stable, "13": spatial, "offline best": best}
        outcome = {}
        for zone, mask in regions.items():
            minimum = max(0., float(np.abs(continuous[mask] - desired[mask]).mean()) - .5)
            bounds = projection_bounds(left[mask], right[mask], origin[mask], desired[mask])
            scores = {key: metrics(target, expected, target, frame, mask) for key, frame in arms.items()}
            for row in scores.values():
                if row["response_mae"] + 1e-10 < minimum or not bounds[0] - 1e-10 <= row["projection"] <= bounds[1] + 1e-10:
                    raise ValueError(f"Observed output violates computed bound: {name}/{zone}")
            outcome[zone] = dict(rounded_mae_lower_bound=minimum, rounded_projection_bounds=bounds,
                                 blocked_by_mae=minimum > 3 + 1e-10,
                                 blocked_by_projection=bounds[1] < .75 - 1e-10 or bounds[0] > 1.25 + 1e-10,
                                 scores=scores)
        cases[name] = dict(group=group, regions=outcome,
                           best_sha=save(out / f"{name}-offline-best.png", best),
                           panel_sha=panel(out / f"{name}-comparison.png",
                                           {"target": {"original": target, "desired edit": expected},
                                            "responses": arms}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=BEYOND, script_sha=digest(script),
            protocol_sha=digest(script.with_name("beyond-edit-response-envelope-protocol.md")), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
