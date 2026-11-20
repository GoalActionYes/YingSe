"""Test the best continuous chroma error with an independent nonnegative gain per pixel."""
import argparse
from pathlib import Path

import numpy as np

from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_portrait_common_chroma_gain import run
from probe_rawtherapee_portrait_cross import score
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import save


def optimum(observed, wanted):
    ratios = np.divide(wanted, observed, out=np.zeros_like(wanted), where=observed != 0)
    candidates = np.concatenate((np.zeros_like(ratios[..., :1]), np.maximum(ratios, 0)), axis=-1)
    errors = np.abs(candidates[..., None] * observed[..., None, :] - wanted[..., None, :]).sum(axis=-1)
    best = np.take_along_axis(candidates, errors.argmin(axis=-1, keepdims=True), axis=-1)
    return best * observed


def evaluate(target, expected, archive, baseline, masks, output):
    if score(target, expected, archive, masks) != baseline:
        raise ValueError('Archived scores changed')
    anchor = chroma(target)
    wanted = chroma(expected) - anchor
    observed = chroma(archive['on']) - chroma(archive['off'])
    change = optimum(observed, wanted)
    lower = np.abs(change - wanted).mean(axis=-1)
    pixels, clipped = render(anchor + change, target @ LUMA)
    scores = score(target, expected, {'off': target, 'on': pixels}, masks)
    minima = {name: float(lower[mask].mean()) for name, mask in masks.items()}
    return {'continuous_l1_lower_bound': minima, 'gamut_fraction': clipped, 'rendered_scores': scores,
            'passed': all(scores[name]['passed'] for name in ('whole', 'neutral')), 'sha256': save(output, pixels)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'original', 'expanded', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()), evaluate_case=evaluate)
