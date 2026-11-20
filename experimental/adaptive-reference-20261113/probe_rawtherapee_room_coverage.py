"""Screen one ambient RAW per archived SyncLight room against the remaining color gap."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_donor_expansion import develop, select
from probe_rawtherapee_exact_color import color_keys, medians
from probe_rawtherapee_nearby_color import estimate
from probe_rawtherapee_portrait_cross import CLI_HASH, PROFILE_HASH, managed, score, section
from probe_rawtherapee_shared_response import STRENGTH
from probe_rawtherapee_support_ceiling import PROFILE_REPORT_SHA, support_mask
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, json_to, save
from render_rawtherapee_sky_pair import check, disabled
from spatial_render import finish_spatial

EXPANSION_SHA = '02fe73af0b652fded98910111caacd1d8640cdcfbc87f9c7347e622288acf912'
SOURCES = {
    'synclight-extra-rooms-input/scene18_1_0_ambient.CR2': '2104a41e84828ada892be4175ab8fdf27ce38261e8fda690ff241ae023d5c307',
    'synclight-extra-rooms-input/scene23_1_0_ambient.DNG': '495bd694f665d8fd205ea9cc305e26ae25f9bd35c655ada1700157832f359feb',
    'synclight-extra-rooms-input/scene36_1_0_ambient.DNG': 'c9eb4bd04016e06e432bab531e344850f180f892b4ed09de6e5cd151b5e90efa',
    'synclight-multi-room-input/scene22_1_0_ambient.DNG': '9116a56b74ecf95ff7c37e6dad9457c491486e11cf1733cdf29783337a68d9a7',
    'synclight-multi-room-input/scene24_1_0_ambient.DNG': '384355f30df71e1135243cf95309c21e5f57248e309bb9708cb04bcc45e19cf2',
    'synclight-multi-room-input/scene37_1_0_ambient.DNG': '7fe30cfb1af92183f296b6f2a5fc9bb7fd313b3685fe507eb30a35d7bfc7a35b',
    'synclight-next-rooms-input/scene1_1_0_ambient.DNG': '85e44826ad53c1ec503a0d0c58485af048dc6a5a3cac9b1c83e41b78cb991d31',
    'synclight-next-rooms-input/scene12_1_0_ambient.CR2': '0d30925154b5c656253ab571824572434950f7465433f7ccd54090bdc4b1143b',
    'synclight-next-rooms-input/scene2_1_0_ambient.DNG': 'b0c530dbccee315b34e460cafeee27e76fcccc0236c2430e924edeef3c5cc4c7',
    'synclight-next-rooms-input/scene21_1_0_ambient.DNG': '4d69f912d3d2439a4ca2fc2108a35e578a99bc67b8fb626c7b202b4400c6a2d8',
    'synclight-real-pixel-input/raw/scene25_1_0_ambient.DNG': 'e04844e21f0f4ab86d82ef9c1fabbfebe8c309d73656641d2025dc14e3f775fb',
    'synclight-real-pixel-input/raw/scene35_1_0_ambient.DNG': '780001c19fc0ab525dbd9728a59fe24e26c32424c15fdfdd4d81f42a34fa383e',
}


def labelling_support(images, query):
    palette = np.unique(np.concatenate([color_keys(rgb) for rgb in images]))
    return estimate(palette, np.zeros((len(palette), 2)), query)[1]


def run(root, cli, response_report, profile_report, expansion_report, out):
    root, cli, out = (path.resolve() for path in (root, cli, out))
    profile = checked_json(profile_report, PROFILE_REPORT_SHA)
    expansion = checked_json(expansion_report, EXPANSION_SHA)
    profile_dir = root / 'rawtherapee-profile-control-v1'
    previous_dir = root / 'rawtherapee-donor-expansion-v1'
    before = managed(profile_dir / 'off/image.png')
    if sha256(before.tobytes()).hexdigest() != profile['input_rgb_sha256']['off']:
        raise ValueError('Frozen original RGB changed')
    previous = checked_rgb(previous_dir / 'expanded-donor.png', expansion['prediction_sha256'])
    missing = ~support_mask(root, root / 'rawtherapee-shared-response-v1', response_report, before)
    prior = []
    for name in expansion['selected']:
        img = managed(previous_dir / f'{name}-off/image.png')
        if sha256(img.tobytes()).hexdigest() != expansion['inputs'][name]['off_rgb_sha256']:
            raise ValueError('Changed saved donor')
        prior.append(img)
    query, inverse = np.unique(color_keys(before[missing]), return_inverse=True)
    added = labelling_support(prior, query)[inverse]
    np.testing.assert_equal(int(added.sum()), expansion['added_pixels'])
    missing[missing] = ~added
    pixels = before[missing]
    query, inverse = np.unique(color_keys(pixels), return_inverse=True)

    check(cli, CLI_HASH)
    author = check(root / 'rawtherapee-portrait-toning-inputs-v1/profile-check.pp3', PROFILE_HASH)
    on = b'\n'.join(section(author, name) for name in (b'Version', b'ColorToning', b'Resize'))
    off = disabled(on, section=b'ColorToning')
    for relative, digest in SOURCES.items():
        check(root / relative, digest)
    out.mkdir()
    inputs, supports, donors = {}, {}, {}
    for relative, digest in SOURCES.items():
        raw = root / relative
        name = raw.stem
        rgb = develop(raw, cli, off, out / f'{name}-off')
        donors[name] = raw, rgb
        supports[name] = labelling_support([rgb], query)[inverse]
        inputs[name] = {'raw_sha256': digest, 'off_rgb_sha256': sha256(rgb.tobytes()).hexdigest(),
                        'covered_pixels': int(supports[name].sum())}

    selected = select(supports)
    union = np.zeros(len(pixels), dtype=bool)
    for name in selected:
        union |= supports[name]
    report = {'expansion_report_sha256': EXPANSION_SHA, 'inputs': inputs, 'selected': selected,
              'missing_pixels': len(pixels), 'candidate_coverage': float(union.mean()),
              'missing_rgb_quantiles': np.quantile(pixels, [.1, .5, .9], axis=0).tolist(),
              'missing_luma_quantiles': np.quantile(pixels @ LUMA, [.1, .5, .9]).tolist(),
              'rendered': False, 'passed': False}
    if report['candidate_coverage'] >= .15:
        keys, changes = [], []
        for name in selected:
            raw, rgb = donors[name]
            edited = develop(raw, cli, on, out / f'{name}-on')
            inputs[name]['on_rgb_sha256'] = sha256(edited.tobytes()).hexdigest()
            expected_donor = finish_spatial(rgb, edited, STRENGTH)
            keys.append(color_keys(rgb))
            changes.append((chroma(expected_donor) - chroma(rgb)).reshape(-1, 2))
        palette, values, _ = medians(np.concatenate(keys), np.concatenate(changes))
        delta, supported = estimate(palette, values, query)
        np.testing.assert_array_equal(supported[inverse], union)
        replacement, _ = render(chroma(pixels[union]) + delta[inverse[union]], pixels[union] @ LUMA)
        filled = missing.copy()
        filled[missing] = union
        picture = previous.copy()
        picture[filled] = replacement
        prediction = save(out / 'room-coverage.png', picture)
        expected = checked_rgb(profile_dir / 'expected.png', profile['output_sha256']['expected'])
        masks = measurement_masks(before, expected, profile['measurement_boxes'])
        rows = score(before, expected, {'off': before, 'on': picture}, {k: v for k, v in masks.items() if v.any()})
        reduction = expansion['scores']['whole']['response_mae'] - rows['whole']['response_mae']
        report.update(rendered=True, prediction_sha256=prediction, scores=rows,
                      passed=rows['whole']['passed'] and reduction >= .25 and
                      all(row['passed'] for row in rows.values() if row['qualified']))
    json_to(out / 'results.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'cli', 'response_report', 'profile_report', 'expansion_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
