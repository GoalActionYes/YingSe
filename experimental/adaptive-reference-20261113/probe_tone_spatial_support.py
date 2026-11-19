"""Balance shadow and highlight reference statistics by spatial support."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_tone_conditioned_reference_histogram import BASE, ARMS, checked_rgb, json_to, load, run, shade_pair

PARENT = "ccdd992b14ee63ac8d0e52f03269364866471c237ddc65ebd5e54eee2b30d2d3"
BLOCK = 32


def support(lower):
    height, width = lower.shape[-2:]
    if height % BLOCK or width % BLOCK:
        raise ValueError("Analysis dimensions must align with spatial blocks")
    tiled = lower.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK)
    dark = tiled.sum(dim=(1, 3), keepdim=True).float()
    bright = BLOCK ** 2 - dark
    shared = torch.minimum(dark, bright)
    fraction = float(shared.sum()) / (height * width)
    metadata = {"shared_fraction": fraction, "shared_blocks": int(torch.count_nonzero(shared))}
    if fraction < .05:
        return None, metadata
    a = tiled * (shared / dark.clamp_min(1))
    b = (~tiled) * (shared / bright.clamp_min(1))
    for weights in (a, b):
        torch.testing.assert_close(weights.sum(dim=(1, 3), keepdim=True), shared, rtol=1e-6, atol=1e-5)
    return (a.reshape_as(lower), b.reshape_as(lower)), metadata


def predict(engine, target, ref, old):
    return shade_pair(engine, target, ref, old, support_weights=support)


def check_balance():
    mask = torch.zeros((1, 1, 32, 64), dtype=torch.bool)
    mask[..., :8, :32] = True
    mask[..., :24, 32:] = True
    weights, report = support(mask)
    assert report == {"shared_fraction": .25, "shared_blocks": 2}
    for part in weights:
        np.testing.assert_allclose([float(part[..., :32].sum()), float(part[..., 32:].sum())],
                                   [256, 256], rtol=1e-6, atol=0)
    assert support(torch.ones_like(mask))[0] is None


def evaluate(out):
    check_balance()
    original = BASE / "tone-conditioned-reference-histogram-v1"
    parent = load(original / "results.json", PARENT)
    run(out, predict=predict)
    report_path = out / "results.json"
    report = json.loads(report_path.read_bytes())
    report["support_parent_sha256"] = PARENT
    counts = {name: 0 for name in ARMS}
    for key, case in report["cases"].items():
        old = parent["cases"][key]
        for role, row in case["roles"].items():
            previous = old["roles"][role]
            if row["scores"]["13"] != previous["scores"]["13"]:
                raise ValueError("Frozen comparison changed")
            row["unbalanced_scores"] = previous["scores"]
            changed_size = {}
            for style, arms in row["hashes"].items():
                for name, sha in arms.items():
                    path = f"{key}-{role}-{style}-{name}.png"
                    before = checked_rgb(original / path, previous["hashes"][style][name])
                    after = checked_rgb(out / path, sha)
                    counts[name] += int(not np.array_equal(before, after))
                    changed_size.setdefault(style, {})[name] = float(np.abs(before.astype(float) - after).mean())
            row["change_vs_unbalanced"] = changed_size
        case["unbalanced_controls"] = old["neutral_controls"]
    report["changed_outputs"] = counts
    json_to(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    evaluate(parser.parse_args().out)
