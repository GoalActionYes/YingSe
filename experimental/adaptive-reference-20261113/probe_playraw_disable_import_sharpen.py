"""Score the same paired portrait operation after disabling import sharpening."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_playraw_rendered_input_operator import (
    FIT_SHA, PARENT_SHA, PROTECTED, REGIONS_SHA, SKIN, SOURCE_SHA, STRENGTH, accepted,
    checked_image, checked_json, chroma, compose, digest, finish_spatial, json_to, panel, read, save, stats,
)

PREVIOUS_SHA = '8d430c18e5d90980cb334ec767029c542b50412c5399d54de7102dddaad5879c'


def score(previous, path, original, wanted, weight, regions):
    observed = {state: read(path / f'{state}.tif', False)[0] for state in ('off', 'on')}
    measured = {state: sha256(frame.tobytes()).hexdigest() for state, frame in observed.items()}
    if sha256(read(previous / 'off.tif', False)[0].tobytes()).hexdigest() != regions['prior_off_sha']:
        raise ValueError('Previous endpoint changed')
    imported = read(previous / 'rendered-input.tif', False)[0]
    np.testing.assert_array_equal(imported, original)
    diff = np.abs(observed['off'].astype(np.int16) - original)
    frame, integrity = compose(original, observed, weight)
    initial = chroma(original)
    actual, expected = chroma(frame) - initial, chroma(wanted) - initial
    scores = {key: stats(actual, expected, zone) for key, zone in regions['masks'].items()}
    no_sharpen = all(not any(token in (path / f'{state}.log').read_text(encoding='utf-8-sig')
                              for token in ('锐化', "processed `sharpen'")) for state in ('off', 'on'))
    base_pass = (int(diff.max()) <= 1 and no_sharpen and integrity['exact_outside'] and integrity['luma_max'] <= .5)
    return dict(off_max=int(diff.max()), off_mae=float(diff.mean()), off_changed=int(np.any(diff, axis=-1).sum()),
                rgb_sha=measured, integrity=integrity, scores=scores, no_sharpen=no_sharpen, base_pass=base_pass,
                sha=save(path / 'candidate.png', frame),
                panel_sha=panel(path / 'comparison.png', {
                    'raw': {'original': original, 'expected': wanted},
                    'rendered input': {'off': observed['off'], 'candidate': frame},
                }))


def run(base, out):
    old = base / 'playraw-rendered-input-operator-v1'
    prior = checked_json(old / 'results.json', PREVIOUS_SHA)
    archived = base / 'playraw-mixed-light-skin-v1/replay-v2'
    receipt = checked_json(archived / 'results.json', SOURCE_SHA)
    before, _ = read(archived / 'off.tif', False)
    after, mask = read(archived / 'on.tif', True)
    if ([sha256(frame.tobytes()).hexdigest() for frame in (before, after)] != receipt['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != receipt['mask_pixels_sha256']):
        raise ValueError('Author inputs changed')
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    areas = {}
    for fold in (0, 1):
        areas[f'high-{fold}'] = (mask >= 204) & (parity != fold)
        areas[f'transition-{fold}'] = (mask > 0) & (mask < 204) & (parity != fold)
    args = dict(prior_off_sha=prior['source']['portable']['rgb_sha']['off'], masks=areas)
    source = score(old / 'mixed', out / 'mixed', before, finish_spatial(before, after, STRENGTH), mask / 255, args)
    controls = checked_json(base / 'playraw-mixed-light-operator-v1/results.json', FIT_SHA)['cases']
    source['passed'] = source['base_pass'] and all(
        accepted(source['scores'][f'high-{fold}']) and source['scores'][f'transition-{fold}']['response_mae']
        <= controls[str(fold)]['arms']['constant']['scores']['transition']['response_mae'] + .25 for fold in (0, 1))

    target_root = base / 'playraw-mixed-light-cross-editor-v2'
    registered = checked_json(Path(__file__).with_name('darktable-cuckoo-legacy-regions.json'), REGIONS_SHA)
    targets = []
    for index, entry in enumerate(checked_json(target_root / 'results.json', PARENT_SHA)['trials']):
        folder = f'cuckoo-{index}'
        before, expected = [checked_image(target_root / str(index), entry, name)
                            for name in ('baseline', 'editor-local')]
        areas = {}
        for name in SKIN + PROTECTED:
            x0, y0, x1, y1 = registered[name]
            area = np.zeros(before.shape[:2], bool)
            area[y0:y1, x0:x1] = True
            areas[name] = area
        active = (areas['cheek'] | areas['chest']).astype(float)
        args = dict(prior_off_sha=prior['targets'][index]['portable']['rgb_sha']['off'], masks=areas)
        value = score(old / folder, out / folder, before, expected, active, args)
        value['passed'] = (value['base_pass'] and all(accepted(value['scores'][key]) for key in SKIN)
                           and all(value['scores'][key]['response_p95'] <= 1.5 for key in PROTECTED))
        targets.append(value)
    protocol = Path(__file__).with_name('playraw-disable-import-sharpen-protocol.md')
    command = Path(__file__).with_name('render_playraw_disable_import_sharpen.ps1')
    json_to(out / 'results.json', dict(previous=PREVIOUS_SHA, protocol_sha=digest(protocol),
            renderer_sha=digest(command), script_sha=digest(Path(__file__)), source=source, targets=targets,
            passed=source['passed'] and all(case['passed'] for case in targets)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('base', 'out'):
        parser.add_argument(label, type=Path)
    run(**vars(parser.parse_args()))
