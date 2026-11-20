"""Compare outputs from matched physical and creative casts in real scenes."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch

from probe_arri_shared_look import metrics
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_transfer import BATCH_CAPTURE_AUDIT, STRENGTH, digest, save, write_json
from probe_multilum_gray_probe import apply, chroma, valid
from probe_multilum_spatial_cast import spatial_dispersion
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze
from spatial_render import SpatialEngine, finish_spatial


SCENES = ('WHITE_green', 'BLUE_red')
LIGHTS = (1, 8)
STATES = ('base', 'real', 'edit')


def prepare_guides(base, real):
    a, b = chroma(base), chroma(real)
    support = valid(base) & valid(real)
    if not support.any():
        raise ValueError('No shared usable pixels')
    gain = np.median((b - a)[support], axis=0)
    edited, clipped = apply(base, gain)
    common = support & valid(edited)
    if not common.any():
        raise ValueError('No shared usable pixels after digital edit')
    observed = np.median((b - a)[common], axis=0)
    authored = np.median((chroma(edited) - a)[common], axis=0)
    strength = float(np.linalg.norm(observed))
    difference = float(np.linalg.norm(observed - authored))
    coverage = float(common.mean())
    qualification = strength >= .08 and difference <= .025 and coverage >= .10 and clipped <= .02
    frames = dict(zip(STATES, (base, real, edited), strict=True))
    cues = {key: spatial_dispersion(frame) for key, frame in frames.items()}
    cue_pass = all(value['tiles'] >= 32 for value in cues.values()) and all(
        cues['real']['dispersion'] - cues[key]['dispersion'] >= .03 for key in ('base', 'edit'))
    return frames, gain, dict(observed_delta=observed.tolist(), edited_delta=authored.tolist(),
                              strength=strength, difference=difference, common_coverage=coverage,
                              clipped_fraction=clipped, qualified=qualification,
                              spatial_cues=cues, spatial_cue_pass=cue_pass and qualification)


def compare(target, expected, ungraded, relit, edited, mask):
    style = metrics(target, expected, ungraded, edited, mask)
    light_mae = float(np.abs(relit[mask].astype(np.int16) - ungraded[mask]).mean())
    between = float(np.abs(relit[mask].astype(np.int16) - edited[mask]).mean())
    return dict(style=style, relit_delta_mae=light_mae, relit_vs_edited_output_mae=between,
                both_pass=style['passed'] and light_mae <= 3)


@torch.inference_mode()
def run(prepared, out):
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    manifest = prepared / 'inputs.json'
    source = json.loads(manifest.read_bytes())
    if source['source_sha256'] != BATCH_CAPTURE_AUDIT or not source['headroom']:
        raise ValueError('Wrong source capture batch or rendering mode')
    records = source['records']
    original = {key: verified(prepared / f'{key}.png', info['image_sha256']) for key, info in records.items()}
    out.mkdir()
    references = {}
    for scene in SCENES:
        frames, gain, audit = prepare_guides(*(original[f'{scene}-{light}'] for light in LIGHTS))
        paths = {state: out / f'{scene}-guide-{state}.png' for state in STATES}
        hashes = {state: save(paths[state], frames[state]) for state in STATES}
        references[scene] = dict(frames=frames, paths=paths, gain=gain, audit=audit | {'sha256': hashes})

    cases = {}
    for scene, donor in permutations(SCENES):
        info = references[donor]
        for light in LIGHTS:
            key = f'{scene}-{light}'
            target = original[key]
            expected, clipped = apply(target, info['gain'])
            expected = finish_spatial(target, expected, STRENGTH)
            expected_path = out / f'{key}-expected.png'
            masks = make_masks(target, records[key]['patches'])
            outputs = {'stable': {}, 'spatial13': {}}
            hashes = {}
            for state in STATES:
                guide = info['frames'][state]
                mapping = analyze(prepared / f'{key}.png', info['paths'][state])
                frames = {'stable': apply_mapping(target, mapping, STRENGTH),
                          'spatial13': finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)}
                for model, frame in frames.items():
                    outputs[model][state] = frame
                    name = f'{key}-{model}-{state}.png'
                    hashes[name] = save(out / name, frame)
            scores = {model: {region: compare(target, expected, *frames.values(), mask)
                              for region, mask in masks.items()} for model, frames in outputs.items()}
            board = out / f'{key}-comparison.png'
            rows = [[(label, frame) for label, frame in
                     [('Fixed target', target), ('Reference base', info['frames']['base']),
                      ('Reference light change', info['frames']['real']),
                      ('Reference digital edit', info['frames']['edit']), ('Expected edit on target', expected)]]]
            rows.extend([[(f'{model} / {state}', frame) for state, frame in frames.items()]
                         for model, frames in outputs.items()])
            cases[key] = dict(reference=donor, reference_qualified=info['audit']['qualified'],
                              authored_target_clipped_fraction=clipped, scores=scores,
                              expected_sha256=save(expected_path, expected), outputs_sha256=hashes,
                              panel_sha256=montage(board, rows))
    write_json(out / 'report.json', dict(prepared_sha256=digest(manifest), checkpoint_sha256=engine.checkpoint_sha,
               stable_fingerprint=algorithm_fingerprint(), strength=STRENGTH,
               references={key: row['audit'] for key, row in references.items()}, cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
