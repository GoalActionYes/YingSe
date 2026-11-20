"""Remove a direction learned only from real relight outputs in another source."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_beyond_rgb_matched_cast import compare
from probe_physical_blue_global_fit import display, linear
from probe_target_luma_edit_mix import BEYOND, LUMA, PARENT, beyond_inputs, mid_inputs
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save
from probe_two_image_chroma_quantile_gain import archive


CONTROL = "two-image-chroma-quantile-gain-v1"


def learn(states):
    moments = []
    for frames in states:
        base, real = (linear(frames[key]) for key in ("base", "real"))
        difference = (real - base).reshape(-1, 3)
        moment = difference.T @ difference
        trace = np.trace(moment)
        if trace <= 0:
            raise ValueError("No learned capture variation")
        moments.append(moment / trace)
    weights, vectors = np.linalg.eigh(np.mean(moments, axis=0))
    return vectors[:, -1], weights


def remove(target, baseline, axis):
    start, current = linear(target), linear(baseline)
    change = current - start
    projected = change - (change @ axis)[..., None] * axis
    norm = np.linalg.norm(change)
    gain = float(np.linalg.norm(projected) / norm) if norm > 0 else 1.
    return display(start + projected), display(start + gain * change), gain


def run(out, exposure_sha):
    root = BASE / CONTROL
    control = checked_json(root / "results.json", exposure_sha)
    directions = {}
    for label, group in (("mit", mid_inputs), ("beyond", beyond_inputs)):
        directions[label] = learn(archive(group, case[0])[1] for case in group())
    out.mkdir()
    cases = {}
    for train, cohort in (("beyond", mid_inputs), ("mit", beyond_inputs)):
        axis, _ = directions[train]
        for name, group, target, expected, regions, _ in cohort():
            guides, original = archive(cohort, name)
            corrected, matched, gains = {}, {}, {}
            for state, frame in original.items():
                corrected[state], matched[state], gains[state] = remove(target, frame, axis)
            same = remove(target, target, axis)
            if any(not np.array_equal(image, target) for image in same[:2]):
                raise ValueError("Identity failed")
            exposure_file = f"{name}-exposure-13.png"
            exposure = checked_rgb(root / exposure_file, control["cases"][name]["hashes"]["exposure-13"])
            altered, naive, _ = remove(target, exposure, axis)
            exposure_delta = [float(np.abs((frame.astype(float) - exposure) @ LUMA).mean())
                              for frame in (altered, naive)]
            scores = {}
            for key, mask in regions.items():
                row = {label: compare(target, expected, *frames.values(), mask) for label, frames in
                       (("13", original), ("projected", corrected), ("matched", matched))}
                for arm in row.values():
                    arm["passed"] = arm["both_pass"] and arm["style"]["off_matches_original_mae"] <= 3
                scores[key] = row
            frames = {f"{arm}-{state}": frame
                      for arm, results in (("projected", corrected), ("matched", matched))
                      for state, frame in results.items()}
            frames.update({"exposure-projected": altered, "exposure-matched": naive})
            cases[name] = dict(group=group, training=train, gains=gains, zones=scores,
                               exposure_luma_error=exposure_delta,
                               hashes={key: save(out / f"{name}-{key}.png", image) for key, image in frames.items()},
                               panel_sha=panel(out / f"{name}-comparison.png",
                                               {"reference": guides, "target": {"original": target, "desired": expected},
                                                "13": original, "projected": corrected, "matched": matched}))
    source = Path(__file__)
    json_to(out / "results.json", dict(parents={"mid": PARENT, "beyond": BEYOND, "exposure": exposure_sha},
            script_sha=digest(source), protocol_sha=digest(source.with_name("cross-source-capture-direction-protocol.md")),
            training={key: dict(axis=axis.tolist(), eigenvalues=values.tolist())
                      for key, (axis, values) in directions.items()}, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("exposure_sha")
    run(**vars(parser.parse_args()))
