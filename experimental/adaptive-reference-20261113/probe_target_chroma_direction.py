"""Find the best constant-luminance edit achievable on a frozen color ray."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_mid_edit_response_envelope import solution
from probe_physical_blue_global_fit import display, linear
from probe_target_linear_luma_edit import preserve_luma
from probe_target_luma_edit_mix import LUMA, beyond_inputs, mid_inputs
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save


PARENT = "b350456d74279e41130e68c2ea979934f3e7962f21d57fd89c8b6a13189bdd41"
MIN_SRGB8_SLOPE = 255 * 1.055 / 2.4


def ray_solution(start, locked, desired):
    delta = locked - start
    direction = delta - (delta @ LUMA)[..., None]
    cap = np.full_like(direction, np.inf)
    np.divide(1 - start, direction, out=cap, where=direction > 0)
    np.divide(-start, direction, out=cap, where=direction < 0)
    upper = cap.min(axis=-1)
    upper[~np.isfinite(upper)] = 0
    endpoint = np.clip(start + upper[..., None] * direction, 0, 1)
    best, _ = solution(start * 255, endpoint * 255, desired * 255)
    best /= 255
    if np.max(np.abs((best - start) @ LUMA)) > 1e-12:
        raise ValueError("Ray used a nonzero luma component")
    return best


def run(out):
    root = BASE / "target-linear-luma-edit-v1"
    report = checked_json(root / "results.json", PARENT)
    out.mkdir()
    cases = {}
    for cohort in (mid_inputs, beyond_inputs):
        for name, group, target, expected, regions, _ in cohort():
            start, wanted = linear(target), linear(expected)
            unrestricted, _ = preserve_luma(target, expected)
            arms = {"direction free": unrestricted}
            bounds = {}
            for model in ("stable", "13"):
                key = f"{model} locked"
                locked = checked_rgb(root / f"{name}-{model}-locked.png", report["cases"][name]["hashes"][key])
                best = ray_solution(start, linear(locked), wanted)
                arms[f"{model} ray"] = display(best)
                arms[key] = locked
                bounds[model] = np.abs(best - wanted).mean(axis=-1) * MIN_SRGB8_SLOPE
            y_gap = np.abs((wanted - start) @ LUMA)
            rows = {}
            for zone, mask in regions.items():
                floors = {key: max(0., float(error[mask].mean()) - .5) for key, error in bounds.items()}
                reading = {key: metrics(target, expected, target, image, mask) for key, image in arms.items()}
                for model, floor in floors.items():
                    if floor > reading[f"{model} ray"]["response_mae"] + 1e-10:
                        raise ValueError("Computed lower bound exceeds measured error")
                rows[zone] = dict(scores=reading, rgb8_mae_lower_bound=floors,
                                  luma_rgb8_mae_lower_bound=max(0., float(y_gap[mask].mean()) *
                                      MIN_SRGB8_SLOPE / (3 * LUMA.max()) - .5),
                                  direction_proven_limiting=[key for key, floor in floors.items()
                                      if floor > 3 + 1e-10 and reading["direction free"]["passed"]])
            hashes = {}
            for key in ("stable ray", "13 ray", "direction free"):
                hashes[key] = save(out / f"{name}-{key.replace(' ', '-')}.png", arms[key])
            cases[name] = dict(group=group, regions=rows, hashes=hashes,
                               panel_sha=panel(out / f"{name}-comparison.png",
                                               {"target": {"original": target, "desired edit": expected},
                                                "response": arms}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=PARENT, script_sha=digest(script),
            protocol_sha=digest(script.with_name("target-chroma-direction-protocol.md")), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
