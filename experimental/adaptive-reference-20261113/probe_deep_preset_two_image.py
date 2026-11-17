"""Run the author's Deep Preset image forward on frozen two-image controls."""
from argparse import ArgumentParser, Namespace
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from background_intervention import region_delta, target_regions
from independent_coco import digest
from prepare_deep_preset_image_weights import OPTIONS
from probe_gray_edge_correction import PARENT_SHA
from probe_psbench_action_rgb_transfer import score
from probe_psbench_teal_portrait import verify_source
from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, decision, projection, read
from probe_tst_reference_spatial_signal import structure_score
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def load_model(assets):
    manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
    for name, sha in manifest['sources'].items():
        if digest(assets / 'networks' / name) != sha:
            raise ValueError('Upstream source hash changed')
    path = assets / 'image-weights.pth'
    if digest(path) != manifest['image_weights_sha256'] or manifest['options'] != OPTIONS:
        raise ValueError('Weight identity changed')
    sys.path.insert(0, str(assets.resolve()))
    architecture = importlib.import_module('networks.network')
    if not Path(architecture.__file__).resolve().is_relative_to(assets.resolve()):
        raise ValueError('Unexpected upstream module')
    with torch.device('meta'):
        model = architecture.get_model('net')(Namespace(**OPTIONS))
    state = torch.load(path, weights_only=True, map_location='cuda')
    keys = model.state_dict()
    expected = {name for name in keys if not name.startswith('llayer_')}
    if set(state) != expected:
        raise ValueError('Incomplete image-forward tensor set')
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected or set(missing) != set(keys) - expected:
        raise ValueError('Unexpected skipped parameters')
    return model.eval(), manifest['image_weights_sha256']


@torch.inference_mode()
def render(model, target, reference):
    if np.array_equal(target, reference):
        return target.copy()
    height, width = target.shape[:2]
    if reference.shape != target.shape:
        reference = np.asarray(Image.fromarray(reference).resize((width, height), Image.Resampling.BICUBIC))
    padding = ((0, -height % 16), (0, -width % 16), (0, 0))

    def tensor(rgb):
        value = np.pad(rgb, padding, mode='edge').astype(np.float64) / 255
        return torch.from_numpy(((value - .5) / .5).astype(np.float32)).permute(2, 0, 1)[None].cuda()

    image, preset, embedding = model.stylize(tensor(target), tensor(reference), False)
    expected_shape = (1, 3, height + padding[0][1], width + padding[1][1])
    if (tuple(image.shape) != expected_shape or preset is not None or embedding is not None
            or not torch.isfinite(image).all()):
        raise ValueError('Unexpected model output')
    encoded = (((image + 1) * .5).clamp(0, 1)[0].permute(1, 2, 0) * 255).cpu().numpy()
    return finish_spatial(target, encoded[:height, :width].astype(np.uint8), .8)


def summarize(origin, predictions, masks):
    details = {}
    for name, values in predictions.items():
        base, candidate = values['baseline'].astype(np.int16), values['candidate'].astype(np.int16)
        delta_base = base - origin['baseline']
        delta_candidate = candidate - origin['candidate']
        details[name] = dict(regions={key: {r: region_delta(rgb, origin[key], mask) for r, mask in masks.items()}
                                      for key, rgb in values.items()},
                             projection={r: projection(delta_candidate, delta_base, mask) for r, mask in masks.items()})
    return details


def run(assets, out):
    manifest = PARENT / 'results.json'
    if digest(manifest) != PARENT_SHA:
        raise ValueError('Unexpected control data')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, weights_sha = load_model(assets)
    out.mkdir(exist_ok=False)
    cases, hashes = [], {}
    for case in data['cases']:
        target = read(case['input_file'])
        pair = case['pair']
        (out / pair).mkdir()
        inputs = {row['name']: row for row in case['variants']}
        predictions = {}
        for name in ('identity', *BACKGROUND, *GLOBAL):
            ref = read(inputs[name]['reference'])
            baseline = read(inputs[name]['methods']['adaptive']['file'])
            rgb = render(model, target, ref)
            path = out / pair / f'{name}.png'
            Image.fromarray(rgb).save(path)
            hashes[path.relative_to(out).as_posix()] = digest(path)
            predictions[name] = dict(baseline=baseline, candidate=rgb)
        mask_name = pair.split('-')[0] + '-human-mask.png'
        mask_path = BASE / 'independent-coco' / mask_name
        if digest(mask_path) != case['parents'][mask_name]:
            raise ValueError('Unexpected scoring mask')
        with Image.open(mask_path) as im:
            human = np.asarray(im.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        masks = target_regions(target, human)
        rows = summarize(predictions['identity'], predictions, masks)
        rows['identity']['geometry'] = structure_score(target, predictions['identity']['candidate'])
        cases.append(dict(pair=pair, variants=rows))

    report = dict(controlled=decision(cases), cases=cases)
    folder = BASE / 'psbench-shared-local-v1'
    evidence = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-psbench-action-pair-evidence.json'
    prior = json.loads(evidence.read_text(encoding='utf-8'))
    if verify_source(folder, '105') != prior['files'] or prior['strength'] != .8:
        raise ValueError('Unexpected diagnostic source')
    source, edited = (decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end'))
    crops = {key: slice(*cols) for key, cols in prior['crop_columns'].items()}
    examples = {}
    for target, reference in [('portrait', 'tower'), ('tower', 'portrait')]:
        origin = source[:, crops[target]]
        path = folder / 'action-pair-v1' / f'{target}.png'
        if digest(path) != prior['predictions'][target]:
            raise ValueError('Unexpected diagnostic baseline')
        rgb = render(model, origin, edited[:, crops[reference]])
        output = out / f'{target}.png'
        Image.fromarray(rgb).save(output)
        hashes[output.name] = digest(output)
        expected = finish_spatial(origin, edited[:, crops[target]], .8).astype(np.int16)
        rows = score(origin, expected, dict(original=origin, current13=decode_image(path), candidate=rgb),
                     np.ones(origin.shape[:2], dtype=bool))
        gate = all(row['methods']['candidate']['mae'] <= min(row['methods']['original']['mae'],
                   row['methods']['current13']['mae'] - 1) for row in rows.values())
        examples[target] = dict(scores=rows, passed=gate and rows['whole']['methods']['candidate']['projection'] > 0)
    report.update(authored=examples, artifacts=hashes, weights_sha256=weights_sha, parent_sha256=PARENT_SHA,
                  pair_sha256=digest(evidence), script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('deep-preset-two-image-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(controlled=report['controlled'], authored=examples)))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
