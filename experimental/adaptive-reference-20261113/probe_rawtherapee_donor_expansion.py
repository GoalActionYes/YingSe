"""Check whether additional verified RAWs fill the known editor color support gap."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_exact_color import color_keys, medians
from probe_rawtherapee_nearby_color import estimate
from probe_rawtherapee_portrait_cross import CLI_HASH, PROFILE_HASH, managed, score, section
from probe_rawtherapee_shared_response import STRENGTH
from probe_rawtherapee_support_ceiling import PROFILE_REPORT_SHA, support_mask
from probe_sparks_shared_chroma import render as render_chroma
from probe_target_mirror_chroma import checked_json, checked_rgb, json_to, save
from render_rawtherapee_sky_pair import check, disabled, render as render_raw
from spatial_render import finish_spatial

SOURCES = {
    'drone-wall-illuminant-input/dining_area.NEF': 'e6576664afa5ad40ab2661be52b1d84f85db34543f44297ebef0d5e94b993c18',
    'drone-wall-illuminant-input/wall_hallway.NEF': '2858d74d1be7948a9255f15827a6b8002fe9332a5e4e8ba05149a3aef8f52cbe',
    'playraw-sky-toggle-v1/DSCF0419.RAF': '9f0af3b458bfdf04cc8de3daa6d3b03c6d776e100c64983191aeff1bf164e4ca',
    'rawtherapee-graduated-sky-v1/1Q8A5461.CR3': '41067ff0b376556d3c79a7fa13111c5b32ae5b14703b0e2c08d6d4ed0a9ef235',
}


def develop(raw, cli, profile, directory):
    directory.mkdir()
    pp3 = directory / 'profile.pp3'
    pp3.write_bytes(profile)
    path = directory / 'image.png'
    env = dict(os.environ, RT_SETTINGS=str(directory / 'settings'), RT_CACHE=str(directory / 'cache'),
               OMP_NUM_THREADS='1')
    render_raw(cli, raw, pp3, path, env)
    return managed(path)


def select(supports):
    selected = []
    uncovered = np.ones_like(next(iter(supports.values())), dtype=bool)
    for _ in range(2):
        remaining = {name: int((mask & uncovered).sum()) for name, mask in supports.items() if name not in selected}
        candidate = max(remaining, key=remaining.get)
        if not remaining[candidate]:
            break
        selected.append(candidate)
        uncovered &= ~supports[candidate]
    return selected


def run(root, cli, response_report, profile_report, out):
    root, cli, out = (path.resolve() for path in (root, cli, out))
    response_dir, profile_dir = root / 'rawtherapee-shared-response-v1', root / 'rawtherapee-profile-control-v1'
    audit = checked_json(profile_report, PROFILE_REPORT_SHA)
    before = managed(profile_dir / 'off/image.png')
    if sha256(before.tobytes()).hexdigest() != audit['input_rgb_sha256']['off']:
        raise ValueError('Frozen working RGB changed')
    previous = checked_rgb(profile_dir / 'nearby.png', audit['output_sha256']['nearby'])
    supported = support_mask(root, response_dir, response_report, before)
    np.testing.assert_allclose(supported.mean(), audit['scores']['nearby']['whole']['coverage'], atol=1e-14, rtol=0)
    missing = ~supported
    missing_keys, inverse = np.unique(color_keys(before[missing]), return_inverse=True)

    check(cli, CLI_HASH)
    author = check(root / 'rawtherapee-portrait-toning-inputs-v1/profile-check.pp3', PROFILE_HASH)
    on = b'\n'.join(section(author, name) for name in (b'Version', b'ColorToning', b'Resize'))
    off = disabled(on, section=b'ColorToning')
    out.mkdir()
    donors, supports, identity = {}, {}, {}
    for path, digest in SOURCES.items():
        raw = root / path
        check(raw, digest)
        name = raw.stem
        baseline = develop(raw, cli, off, out / f'{name}-off')
        donors[name] = raw, baseline
        palette = np.unique(color_keys(baseline))
        _, close = estimate(palette, np.zeros((len(palette), 2)), missing_keys)
        supports[name] = close[inverse]
        identity[name] = {'raw_sha256': digest, 'off_rgb_sha256': sha256(baseline.tobytes()).hexdigest(),
                          'missing_pixels_supported': int(supports[name].sum())}

    selected = select(supports)
    picture = previous.copy()
    filled = np.zeros_like(missing)
    if selected:
        keys, changes = [], []
        for name in selected:
            raw, baseline = donors[name]
            authored = develop(raw, cli, on, out / f'{name}-on')
            identity[name]['on_rgb_sha256'] = sha256(authored.tobytes()).hexdigest()
            keys.append(color_keys(baseline))
            edited = finish_spatial(baseline, authored, STRENGTH)
            changes.append((chroma(edited) - chroma(baseline)).reshape(-1, 2))
        palette, delta, _ = medians(np.concatenate(keys), np.concatenate(changes))
        predicted, added = estimate(palette, delta, missing_keys)
        filled[missing] = added[inverse]
        pixels = before[filled]
        rgb, _ = render_chroma(chroma(pixels) + predicted[inverse][added[inverse]], pixels @ LUMA)
        picture[filled] = rgb
    np.testing.assert_array_equal(picture[~filled], previous[~filled])
    output = save(out / 'expanded-donor.png', picture)

    expected = checked_rgb(profile_dir / 'expected.png', audit['output_sha256']['expected'])
    masks = measurement_masks(before, expected, audit['measurement_boxes'])
    masks = {label: mask for label, mask in masks.items() if mask.any()}
    results = score(before, expected, {'off': before, 'on': picture}, masks)
    energy = np.square(chroma(expected) - chroma(before)).sum(axis=-1)
    improvement = audit['scores']['nearby']['whole']['response_mae'] - results['whole']['response_mae']
    passed = results['whole']['passed'] and all(row['passed'] for row in results.values() if row['qualified'])
    json_to(out / 'results.json', {'profile_report_sha256': PROFILE_REPORT_SHA, 'inputs': identity,
                                  'selected': selected, 'added_pixels': int(filled.sum()),
                                  'previous_missing_pixels': int(missing.sum()),
                                  'supported_missing_energy_share': float(energy[filled].sum() / energy[missing].sum()),
                                  'scores': results, 'mae_improvement': improvement,
                                  'passed': passed and improvement >= .25, 'prediction_sha256': output})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'cli', 'response_report', 'profile_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
