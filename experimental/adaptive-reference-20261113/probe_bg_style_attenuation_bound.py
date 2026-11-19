"""Find the style-constrained optimum of a constant color-direction suppression."""

import argparse
from pathlib import Path

import numpy as np

from probe_bg_style_direction_overlap import A_SHA, B_SHA, remove_axis
from probe_hdm_hdr_display import json_to
from probe_playraw_mixed_light_transfer import LUMA, chroma, stats
from probe_reference_skin_chroma_attribution import regions
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, save


OVERLAP_SHA = 'd7ea98c99217d35471426bf291ee1d8d654cdf79d6eb7a06aaa00ddafc94a859'


def minimum_anchor(delta, component):
    weight = np.abs(component)
    take = weight > 1e-12
    if not take.any():
        return 0.
    ratios = delta[take] / component[take]
    order = np.argsort(ratios)
    total = np.cumsum(weight[take][order])
    return float(np.clip(ratios[order[np.searchsorted(total, total[-1]/2)]], 0, 1))


def first_allowed(delta, component, optimum):
    if np.abs(delta).mean() <= 1.5:
        return 0.
    if np.abs(delta - optimum*component).mean() > 1.5:
        return None
    low, high = 0., optimum
    for _ in range(40):
        middle = (low + high) / 2
        if np.abs(delta - middle*component).mean() <= 1.5:
            high = middle
        else:
            low = middle
    return high


def examine(a, b, c, f, old, path, label):
    axis = np.asarray(old['axis'])
    style = chroma(b) - chroma(a)
    nuisance = chroma(f) - chroma(c)
    s = style - remove_axis(style, axis)
    n = nuisance - remove_axis(nuisance, axis)
    aligned = 1 - old['continuous_style_projection']
    per_unit_style_error = np.abs(s).mean()
    cap = float(min(1., .25/aligned if aligned > 0 else 1., 3/per_unit_style_error if per_unit_style_error else 1.))
    optimum = minimum_anchor(nuisance, n)
    first = first_allowed(nuisance, n, optimum)
    alpha = min(optimum, cap)
    residual = float(np.abs(nuisance - alpha*n).mean())
    style_out, clip1 = render(chroma(a) + style - alpha*s, b @ LUMA)
    bg_out, clip2 = render(chroma(c) + nuisance - alpha*n, f @ LUMA)
    flat_actual = (chroma(style_out) - chroma(a)).reshape(-1, 2)
    flat_want = style.reshape(-1, 2)
    colors = stats(flat_actual, flat_want, np.ones(len(flat_want), dtype=bool))
    drift = float(np.abs(chroma(bg_out) - chroma(c)).mean())
    if first is not None and (first <= cap) != (residual <= 1.5):
        raise ValueError('Inconsistent feasibility computation')
    results = {'eligible': old['eligible'], 'max_style_attenuation': cap, 'min_bg_attenuation': first,
               'attenuation': alpha, 'continuous_bg': residual, 'continuous_style_projection': 1 - alpha*aligned,
               'continuous_pass': residual <= 1.5, 'pixel_bg': drift, 'pixel_style': colors,
               'pixel_pass': drift <= 1.5 and .75 <= colors['projection'] <= 1.25 and colors['response_mae'] <= 3,
               'clipped_fractions': [clip1, clip2],
               'images': {token: save(path / f'{label}-{token}.png', rgb)
                          for token, rgb in [('style', style_out), ('background', bg_out)]}}
    return results


def run(root, out):
    ar, br = root / 'bg-real-two-look-switch-v3', root / 'bg-real-film-b-qualification-v1'
    overlap = checked_json(root / 'bg-style-direction-overlap-v1/results.json', OVERLAP_SHA)
    aa, bb = checked_json(ar / 'results.json', A_SHA), checked_json(br / 'results.json', B_SHA)
    out.mkdir()
    rows = {}
    for key, boxes in regions().items():
        local = out / key
        local.mkdir()
        a = checked_rgb(ar / key / 'expected-a.png', aa['results']['13'][key]['asset_hashes']['expected-a'])
        b = checked_rgb(br / f'{key}-expected.png', bb['qualification'][key]['expected_sha256'])
        arms = {}
        for arm in ('13', 'expanded'):
            files = bb['comparison'][arm]['cases'][key]['outputs']
            c, f = [checked_rgb(br / f'{arm}-{key}-{role}-off.png', files[f'{role}-off']) for role in ('C1', 'F1')]
            arms[arm] = {name: examine(a[area], b[area], c[area], f[area], overlap['cases'][key][arm][name],
                                       local, f'{arm}-{i}')
                         for i, (name, area) in enumerate({'whole': np.s_[:, :], **boxes}.items())}
        rows[key] = arms
    json_to(out / 'results.json', {'parent': OVERLAP_SHA, 'cases': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
