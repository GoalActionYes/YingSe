"""Replay frozen edit responses under a declared constant-linear-luma edit."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_physical_blue_global_fit import display, linear
from probe_target_luma_edit_mix import BEYOND, LUMA, PARENT, beyond_inputs, mid_inputs
from probe_target_mirror_chroma import digest, json_to, panel, save


def preserve_luma(target, candidate):
    initial = linear(target)
    changed = linear(candidate)
    y = initial @ LUMA
    observed = changed @ LUMA
    ratio = np.divide(y, observed, out=np.zeros_like(y), where=observed > 0)
    adjusted = np.where(observed[..., None] > 0, changed * ratio[..., None], y[..., None])
    peak = adjusted.max(axis=-1)
    headroom = np.divide(1 - y, peak - y, out=np.ones_like(y), where=peak > 1)
    constrained = y[..., None] + (adjusted - y[..., None]) * headroom[..., None]
    error = float(np.max(np.abs(constrained @ LUMA - y)))
    if error > 1e-12 or constrained.min() < -1e-12 or constrained.max() > 1 + 1e-12:
        raise ValueError("Linear luma or gamut invariant failed")
    return display(constrained), dict(max_linear_luma_error=error, compressed_fraction=float((peak > 1).mean()))


def run(out):
    out.mkdir()
    cases = {}
    for cohort in (mid_inputs, beyond_inputs):
        for name, group, target, expected, regions, edits in cohort():
            held, _ = preserve_luma(target, target)
            if not np.array_equal(held, target):
                raise ValueError("Identity invariant failed")
            arms = {}
            audit = {}
            for model, edit in zip(("stable", "13"), edits, strict=True):
                base = np.clip(target + edit, 0, 255).astype(np.uint8)
                locked, audit[model] = preserve_luma(target, base)
                arms.update({model: base, f"{model} locked": locked})
            scores = {zone: {key: metrics(target, expected, target, frame, mask)
                             for key, frame in arms.items()} for zone, mask in regions.items()}
            regressed = {}
            for model in audit:
                regressed[model] = [
                    zone for zone, record in scores.items()
                    if record[model]["passed"] and (
                        not record[f"{model} locked"]["passed"] or
                        record[f"{model} locked"]["response_mae"] > record[model]["response_mae"] + 1)
                ]
            cases[name] = dict(group=group, audit=audit, regions=scores, regressed=regressed,
                               hashes={key: save(out / f"{name}-{key.replace(' ', '-')}.png", frame)
                                       for key, frame in arms.items()},
                               panel_sha=panel(out / f"{name}-comparison.png",
                                               {"target": {"original": target, "desired edit": expected},
                                                "response": arms}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parents={"mid": PARENT, "beyond": BEYOND}, script_sha=digest(script),
            protocol_sha=digest(script.with_name("target-linear-luma-edit-protocol.md")), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
