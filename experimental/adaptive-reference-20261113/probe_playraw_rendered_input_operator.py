"""Apply a recorded RAW editor operation to tagged sRGB target pixels."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from probe_playraw_cross_editor_gain import PARENT_SHA, checked_image
from probe_playraw_mixed_light_cross_editor import FIT_SHA, INPUTS, PROTECTED, REGIONS_SHA, SKIN, default_states
from probe_playraw_mixed_light_skin import export
from probe_playraw_mixed_light_transfer import LUMA, PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, digest, json_to, panel, save
from spatial_render import finish_spatial


def check_input(base, name):
    relative, expected = INPUTS[name]
    path = base / relative
    if digest(path) != expected:
        raise ValueError(f'Unexpected frozen {name} file')
    return path


def endpoints(before, cli, settings, folder):
    folder.mkdir()
    input_file = folder / 'rendered-input.tif'
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(before).save(input_file, icc_profile=profile)
    editor_env = {key: value for key, value in os.environ.items() if key != 'OMP_NUM_THREADS'}
    states = {}
    for label, xmp_bytes in zip(('off', 'on'), settings, strict=True):
        xmp = folder / f'{label}.xmp'
        xmp.write_bytes(xmp_bytes)
        pixels, _ = read(export(cli, input_file, xmp, folder, label, False, env=editor_env), False)
        if pixels.shape != before.shape:
            raise ValueError(f'Shape changed for {folder.name}/{label}')
        states[label] = pixels
    deviation = np.abs(states['off'].astype(np.int16) - before)
    return states, dict(off_max=int(deviation.max()), off_mae=float(deviation.mean()),
                        off_changed_pixels=int(np.any(deviation, axis=-1).sum()),
                        rgb_sha={tag: sha256(rgb.tobytes()).hexdigest() for tag, rgb in states.items()})


def compose(before, states, weight):
    active = weight > 0
    change = chroma(states['on'][active]) - chroma(states['off'][active])
    initial = before[active] @ LUMA
    result = before.copy()
    result[active], gamut = render(chroma(before[active]) + (STRENGTH * weight[active, None]) * change, initial)
    return result, dict(gamut=gamut, luma_max=float(np.abs(result[active] @ LUMA - initial).max()),
                        exact_outside=bool(np.array_equal(result[~active], before[~active])))


def accepted(row):
    return .75 <= row['projection'] <= 1.25 and row['response_mae'] <= 3


def evaluate(before, expected, weight, zones, cli, settings, folder):
    states, portable = endpoints(before, cli, settings, folder)
    candidate, integrity = compose(before, states, weight)
    baseline = chroma(before)
    actual, wanted = chroma(candidate) - baseline, chroma(expected) - baseline
    analysis = {name: stats(actual, wanted, region) for name, region in zones.items()}
    receipt = dict(portable=portable, integrity=integrity, scores=analysis,
                   sha=save(folder / 'candidate.png', candidate),
                   panel_sha=panel(folder / 'comparison.png', {
                       'RAW operation': {'off': before, 'expected': expected},
                       'rendered input': {'CLI off': states['off'], 'candidate': candidate},
                   }))
    return receipt, portable['off_max'] <= 1 and integrity['exact_outside'] and integrity['luma_max'] <= .5


def target(base, rectangles, info, cli, settings, folder, repeat):
    old = base / 'playraw-mixed-light-cross-editor-v2' / str(repeat)
    before, expected = [checked_image(old, info, tag) for tag in ('baseline', 'editor-local')]
    masks = {}
    for name in SKIN + PROTECTED:
        x0, y0, x1, y1 = rectangles[name]
        mask = np.zeros(before.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks[name] = mask
    weight = (masks['cheek'] | masks['chest']).astype(float)
    result, faithful = evaluate(before, expected, weight, masks, cli, settings, folder)
    result['passed'] = (faithful and all(accepted(result['scores'][key]) for key in SKIN)
                        and all(result['scores'][key]['response_p95'] <= 1.5 for key in PROTECTED))
    return result


def original(base, cli, settings, folder):
    archive = base / 'playraw-mixed-light-skin-v1/replay-v2'
    data = checked_json(archive / 'results.json', SOURCE_SHA)
    before, _ = read(archive / 'off.tif', False)
    after, mask = read(archive / 'on.tif', True)
    if ([sha256(frame.tobytes()).hexdigest() for frame in (before, after)] != data['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != data['mask_pixels_sha256']):
        raise ValueError('Changed author RGB or mask')
    expected = finish_spatial(before, after, STRENGTH)
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    folds = checked_json(base / 'playraw-mixed-light-operator-v1/results.json', FIT_SHA)['cases']
    zones = {}
    for fold in (0, 1):
        zones[f'high-{fold}'] = (parity != fold) & (mask >= 204)
        zones[f'transition-{fold}'] = (parity != fold) & (mask > 0) & (mask < 204)
    result, faithful = evaluate(before, expected, mask.astype(float) / 255, zones, cli, settings, folder)
    result['passed'] = faithful and all(
        accepted(result['scores'][f'high-{fold}'])
        and result['scores'][f'transition-{fold}']['response_mae']
        <= folds[str(fold)]['arms']['constant']['scores']['transition']['response_mae'] + .25
        for fold in (0, 1))
    return result


def run(base, out):
    if any(not path.is_absolute() or not path.as_posix().isascii() for path in (base, out)):
        raise ValueError('Use absolute ASCII paths for the editor')
    cli = check_input(base, 'cli')
    settings = default_states(check_input(base, 'author').read_bytes())
    parent = checked_json(base / 'playraw-mixed-light-cross-editor-v2/results.json', PARENT_SHA)
    rectangles = checked_json(Path(__file__).with_name('darktable-cuckoo-legacy-regions.json'), REGIONS_SHA)
    out.mkdir()
    source = original(base, cli, settings, out / 'mixed')
    targets = [target(base, rectangles, item, cli, settings, out / f'cuckoo-{i}', i)
               for i, item in enumerate(parent['trials'])]
    json_to(out / 'results.json', dict(
        sources=[SOURCE_SHA, FIT_SHA, PARENT_SHA], cli_sha=INPUTS['cli'][1], author_sha=INPUTS['author'][1],
        script_sha=digest(Path(__file__)), settings_sha=[sha256(setting).hexdigest() for setting in settings],
        protocol_sha=digest(Path(__file__).with_name('playraw-rendered-input-operator-protocol.md')),
        source=source, targets=targets, passed=source['passed'] and all(item['passed'] for item in targets)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
