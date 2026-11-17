"""Compare a fixed hue substitution using the two saved two-image models."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import LUMA, target_regions
from independent_coco import digest
from probe_deep_preset_two_image import summarize
from probe_gray_edge_correction import PARENT_SHA
from probe_psbench_action_rgb_transfer import score
from probe_psbench_teal_portrait import verify_source
from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, decision, read
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def hue_compose(primary, donor):
    if (primary.shape != donor.shape or primary.dtype != np.uint8 or donor.dtype != np.uint8
            or primary.ndim != 3 or primary.shape[2] != 3):
        raise ValueError('Expected equally shaped RGB8 model outputs')
    light = (primary @ LUMA)[..., None]
    primary_chroma = primary - light
    donor_chroma = donor - (donor @ LUMA)[..., None]
    length = np.linalg.norm(primary_chroma, axis=-1, keepdims=True)
    donor_length = np.linalg.norm(donor_chroma, axis=-1, keepdims=True)
    applicable = donor_length >= 2
    factor = np.divide(length, donor_length, out=np.zeros_like(length), where=applicable)
    shift = np.where(applicable, donor_chroma * factor, primary_chroma)
    bounds = np.divide(np.where(shift < 0, -light, 255 - light), shift,
                       out=np.full_like(shift, np.inf), where=shift != 0)
    gamut_scale = np.minimum(1., bounds.min(axis=-1, keepdims=True)).clip(0, 1)
    output = np.rint((light + shift * gamut_scale).clip(0, 255)).astype(np.uint8)
    new_light = (output @ LUMA)[..., None]
    new_length = np.linalg.norm(output - new_light, axis=-1, keepdims=True)
    diagnostics = dict(used_donor_fraction=float(applicable.mean()),
                       limited_fraction=float(np.mean(gamut_scale < 1 - 1e-10)),
                       mean_luma_error=float(np.abs(new_light - light).mean()),
                       max_luma_error=float(np.abs(new_light - light).max()),
                       mean_chroma_amplitude_error=float(np.abs(new_length - length).mean()))
    return output, diagnostics


def run(deep_outputs, deep_evidence, out):
    published = json.loads(deep_evidence.read_text(encoding='utf-8'))
    parent = PARENT / 'results.json'
    if digest(parent) != PARENT_SHA or published['parent_sha256'] != PARENT_SHA:
        raise ValueError('Unexpected controlled baseline')
    data = json.loads(parent.read_text(encoding='utf-8'))
    out.mkdir(exist_ok=False)
    cases, artifacts, diagnostics = [], {}, {}
    for case in data['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        (out / pair).mkdir()
        inputs = {row['name']: row for row in case['variants']}
        predictions = {}
        for name in ('identity', *BACKGROUND, *GLOBAL):
            key = f'{pair}/{name}.png'
            donor = deep_outputs / key
            if digest(donor) != published['artifacts'][key]:
                raise ValueError(f'Unexpected donor output: {key}')
            base = read(inputs[name]['methods']['adaptive']['file'])
            rgb, metrics = hue_compose(base, decode_image(donor))
            Image.fromarray(rgb).save(out / key)
            artifacts[key] = digest(out / key)
            diagnostics[key] = metrics
            predictions[name] = dict(baseline=base, candidate=rgb)
        mask_name = pair.split('-')[0] + '-human-mask.png'
        mask_path = BASE / 'independent-coco' / mask_name
        if digest(mask_path) != case['parents'][mask_name]:
            raise ValueError('Unexpected scoring mask')
        with Image.open(mask_path) as image:
            human = np.asarray(image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        cases.append(dict(pair=pair, variants=summarize(predictions['identity'], predictions, target_regions(target, human))))

    folder = BASE / 'psbench-shared-local-v1'
    path = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-psbench-action-pair-evidence.json'
    pair = json.loads(path.read_text(encoding='utf-8'))
    if verify_source(folder, '105') != pair['files'] or digest(path) != published['pair_sha256']:
        raise ValueError('Unexpected PSBench sample')
    source, edited = (decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end'))
    comparisons = {}
    for name, (left, right) in pair['crop_columns'].items():
        origin = source[:, left:right]
        baseline = folder / 'action-pair-v1' / f'{name}.png'
        donor = deep_outputs / f'{name}.png'
        if digest(baseline) != pair['predictions'][name] or digest(donor) != published['artifacts'][donor.name]:
            raise ValueError(f'Unexpected source outputs: {name}')
        current, deep = decode_image(baseline), decode_image(donor)
        candidate, debug = hue_compose(current, deep)
        output = out / donor.name
        Image.fromarray(candidate).save(output)
        artifacts[output.name] = digest(output)
        diagnostics[output.name] = debug
        expected = finish_spatial(origin, edited[:, left:right], .8).astype(np.int16)
        rows = score(origin, expected, dict(original=origin, current13=current, deep_preset=deep, candidate=candidate),
                     np.ones(origin.shape[:2], dtype=bool))
        gate = all(row['methods']['candidate']['mae'] <= min(row['methods']['original']['mae'],
                   row['methods']['current13']['mae'] - 1) for row in rows.values())
        comparisons[name] = dict(scores=rows, passed=gate and rows['whole']['methods']['candidate']['projection'] > 0)
    report = dict(controlled=decision(cases), authored=comparisons, cases=cases, artifacts=artifacts,
                  composition=diagnostics, deep_preset_sha256=digest(deep_evidence), parent_sha256=PARENT_SHA,
                  script_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'deep-preset-hue-composition-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(controlled=report['controlled'], authored=comparisons)))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('deep_outputs', 'deep_evidence', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
