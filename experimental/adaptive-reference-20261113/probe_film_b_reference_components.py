"""Separate measured Film B luminance and chroma signals in the reference."""
import argparse
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints, verified
from probe_arri_shared_look import metrics
from probe_arri_two_looks import ASSIGNMENTS
from probe_target_mirror_chroma import BASE, CHECKPOINT, LUMA, STRENGTH, SpatialEngine, chroma, json_to, panel, render, save
from spatial_render import finish_spatial
from verify_development_inputs import MANIFESTS, manifest


def grade(engine, target, reference):
    return finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)


def combine(before, after):
    old_y, new_y = before @ LUMA, after @ LUMA
    return {
        'chroma': render(chroma(after, new_y), old_y),
        'luma': render(chroma(before, old_y), new_y),
    }


def measure(engine, out, original, styled, reference, prefix, receipt):
    truth = finish_spatial(original, styled, STRENGTH)
    source, edit = reference
    root = BASE / 'arri-film-b-crossed-v1'
    off, on = [verified(root / f'{prefix}-{state}.png', receipt['output_sha256'][state])
               for state in ('off', 'film_b')]
    for inputs, stored in ((source, off), (edit, on)):
        np.testing.assert_array_equal(grade(engine, original, inputs), stored)
    outputs, artifacts, board = {'full': on}, {}, {}
    for name, (mixed, clipped) in combine(source, edit).items():
        outputs[name] = grade(engine, original, mixed)
        artifacts[name] = dict(reference_sha256=save(out / f'{prefix}-{name}-reference.png', mixed),
                               output_sha256=save(out / f'{prefix}-{name}.png', outputs[name]), clipped=clipped)
        board[name] = {'reference': mixed, 'target': original, '13': outputs[name], 'known edit': truth}
    whole = np.ones(original.shape[:2], dtype=bool)
    neutral = np.logical_and(*(target_regions(x, ~whole)['neutral_proxy'] for x in (original, styled)))
    if neutral.sum() < 256:
        raise ValueError('Lost common neutral support')
    results = {tag: {name: metrics(original, truth, off, frame, mask) for name, frame in outputs.items()}
               for tag, mask in (('whole', whole), ('neutral', neutral))}
    interaction = on.astype(float) + off - outputs['chroma'] - outputs['luma']
    extra = {'whole': float(np.abs(interaction).mean()), 'neutral': float(np.abs(interaction[neutral]).mean())}
    for tag, values in results.items():
        if values['full'] != receipt[tag]:
            raise ValueError('Changed full edit score')
    return dict(scores=results, interaction_rgb8=extra, artifacts=artifacts,
                panel_sha256=panel(out / f'{prefix}-comparison.png', board))


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    source = 'arri-two-looks-v2'
    known = manifest(BASE, source)
    baseline = manifest(BASE, 'arri-film-b-crossed-v1')
    if engine.checkpoint_sha != CHECKPOINT or baseline['checkpoint_sha256'] != CHECKPOINT or baseline['source_sha256'] != MANIFESTS[source]:
        raise ValueError('Changed frozen control')
    scenes = {name: endpoints(BASE / source, known, name) for name in ASSIGNMENTS}
    out.mkdir()
    cases = {}
    for name, receipt in baseline['cases'].items():
        scene, donor = receipt['target'], receipt['reference']
        cases[name] = measure(engine, out, *scenes[scene], scenes[donor], f'{scene}-from-{donor}', receipt)
    weld = [row for name,row in cases.items() if name.startswith('sparks_4592<-')]
    if len(cases) != 6 or len(weld) != 2:
        raise ValueError('Changed direction set')
    scores = {arm: [row['scores'][area][arm]['projection'] for row in weld for area in ('whole', 'neutral')]
              for arm in ('luma', 'chroma')}
    leading = [a for a,b in (('luma','chroma'),('chroma','luma'))
               if max(scores[a]) < 0 <= min(scores[b])]
    json_to(out / 'results.json', dict(cases=cases, checkpoint_sha256=CHECKPOINT,
                                      welder_projections=scores, isolated=leading[0] if leading else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
