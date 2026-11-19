"""Measure loss of real style when a dominant background-drift direction is removed."""

import argparse
from pathlib import Path

import numpy as np

from probe_hdm_hdr_display import json_to
from probe_playraw_mixed_light_transfer import LUMA, chroma, stats
from probe_reference_skin_chroma_attribution import regions
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, save


A_SHA = 'e1ee9194d2117910264631804536485be059b204c6ce94d31e0f2fdbab26fd81'
B_SHA = '881bf466bd4456ba5b6a4ffe5a3126ddb7fb3de25912291a5790b9fb83813e16'


def remove_axis(change, axis):
    return change - (change @ axis)[..., None] * axis


def measure(a, b, off_c, off_f, path):
    nuisance = (chroma(off_f) - chroma(off_c)).reshape(-1, 2)
    style = chroma(b) - chroma(a)
    values, vectors = np.linalg.eigh(nuisance.T @ nuisance)
    axis = vectors[:, -1]
    support = float(values[-1] / values.sum()) if values.sum() else 0.
    strength = float(np.abs(nuisance).mean())
    desired_strength = float(np.abs(style).mean())
    allowed = remove_axis(style, axis)
    result, clipped = render(chroma(a) + allowed, b @ LUMA)
    actual = (chroma(result) - chroma(a)).reshape(-1, 2)
    target = style.reshape(-1, 2)
    metrics = stats(actual, target, np.ones(len(actual), dtype=bool))
    energy = np.sum(target**2)
    retained = float(np.sum(allowed*style) / energy) if energy else None
    left = float(np.abs(remove_axis(nuisance, axis)).mean())
    eligible = len(nuisance) >= 128 and strength >= 1 and support >= .8 and desired_strength >= 1
    return {'nuisance_mae': strength, 'nuisance_energy_on_axis': support, 'axis': axis.tolist(),
            'expected_style_mae': desired_strength, 'nuisance_residual_mae': left,
            'continuous_style_projection': retained, 'rendered': metrics, 'eligible': eligible,
            'directional_conflict': eligible and retained < .75,
            'joint_pass': eligible and .75 <= metrics['projection'] <= 1.25 and metrics['response_mae'] <= 3 and left <= 1.5,
            'clipped_fraction': clipped, 'sha256': save(path, result)}


def run(root, out):
    ar, br = root / 'bg-real-two-look-switch-v3', root / 'bg-real-film-b-qualification-v1'
    info = checked_json(ar / 'results.json', A_SHA)
    older = checked_json(br / 'results.json', B_SHA)
    boxes = regions()
    out.mkdir()
    cases = {}
    for key, registered in boxes.items():
        photo_dir = out / key
        photo_dir.mkdir()
        row_a = info['results']['13'][key]['asset_hashes']
        a = checked_rgb(ar / key / 'expected-a.png', row_a['expected-a'])
        b = checked_rgb(br / f'{key}-expected.png', older['qualification'][key]['expected_sha256'])
        shapes = {'whole': np.s_[:, :], **registered}
        groups = {}
        for arm in ('13', 'expanded'):
            old = older['comparison'][arm]['cases'][key]['outputs']
            c, f = [checked_rgb(br / f'{arm}-{key}-{role}-off.png', old[f'{role}-off']) for role in ('C1', 'F1')]
            groups[arm] = {label: measure(a[area], b[area], c[area], f[area], photo_dir / f'{arm}-{index}.png')
                           for index, (label, area) in enumerate(shapes.items())}
        cases[key] = groups
    json_to(out / 'results.json', {'parents': [A_SHA, B_SHA], 'cases': cases})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
