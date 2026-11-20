"""Identify the role of reference luma in the matched wall-cast control."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_bg_real_channel_replay import histogram, infer
from probe_mid_matched_wall_cast import PARENT as SOURCE, compare, targets
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, LUMA, SpatialEngine, checked_json, checked_rgb, digest, json_to, panel, save,
)

PARENT = '9f1ca17f3b643017ec42f0a0003184ac3da455e5f93c40f93fe9c02669a0811a'
CHANNELS = 128


def transfer_luma(recipient, donor):
    result = torch.cat((donor[:, :1], recipient[:, 1:]), dim=1)
    torch.testing.assert_close(result[:, 1:], recipient[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(result[:, :1], donor[:, :1], atol=0, rtol=0)
    return result


def difference(first, second, regions):
    delta = first.astype(np.int16) - second
    return {key: dict(rgb=float(np.abs(delta[area]).mean()), y=float(np.abs(delta[area] @ LUMA).mean()),
                      chroma=float(np.abs(delta[area] - (delta[area] @ LUMA)[:, None]).mean()))
            for key, area in regions.items()}


def run(out):
    torch.set_num_threads(4)
    parent = BASE / 'mid-matched-wall-cast-v1'
    previous = checked_json(parent / 'results.json', PARENT)
    ledger = checked_json(BASE / 'mid-gray-anchor-v1.json', SOURCE)
    if previous['spatial'] != CHECKPOINT or not previous['reference']['qualified']:
        raise ValueError('Changed model or reference qualification')
    model = SpatialEngine('multiref-v1')
    if model.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed checkpoint')
    guides = {label: checked_rgb(parent / f'reference-{label}.png', key)
              for label, key in previous['inputs'].items()}
    hist = {name: histogram(model, photo) for name, photo in guides.items()}
    if any(value.shape != (1, 3, CHANNELS) for value in hist.values()):
        raise ValueError('Changed histogram layout')
    trials = {
        'captured_chroma_edited_y': transfer_luma(hist['captured'], hist['edited']),
        'edited_chroma_captured_y': transfer_luma(hist['edited'], hist['captured']),
    }
    out.mkdir()
    results = {}
    for name, (target, regions) in targets(ledger['scenes']).items():
        receipt = previous['cases'][name]
        np.testing.assert_array_equal(target, checked_rgb(parent / f'{name}-target.png', receipt['target_sha']))
        expected = checked_rgb(parent / f'{name}-expected.png', receipt['expected_sha'])
        ref_target = histogram(model, target)
        low = model.api['TF'].resize(model.analysis(target), (64, 64), antialias=True)
        old, frames = {}, {}
        for label, key in receipt['models']['13']['hashes'].items():
            old[label] = checked_rgb(parent / f'{name}-13-{label}.png', key)
            rendered = infer(model, target, low, ref_target, hist[label])
            np.testing.assert_array_equal(rendered, old[label])
        for label, stats in trials.items():
            frames[label] = infer(model, target, low, ref_target, stats)
        corrected = frames['captured_chroma_edited_y']
        reverse = frames['edited_chroma_captured_y']
        scores = {key: compare(target, expected, old['base'], corrected, old['edited'], area)
                  for key, area in regions.items()}
        results[name] = dict(
            common_y=scores, passed=all(row['both_pass'] and row['style']['off_matches_original_mae'] <= 3
                                        for row in scores.values()),
            pairs={
                'change_y_with_captured_chroma': difference(corrected, old['captured'], regions),
                'change_y_with_edited_chroma': difference(reverse, old['edited'], regions),
                'change_chroma_with_edited_y': difference(corrected, old['edited'], regions),
            },
            hashes={label: save(out / f'{name}-{label}.png', frame) for label, frame in frames.items()},
            panel_sha=panel(out / f'{name}-comparison.png', {
                'target': {'original': target, 'intended edit': expected}, 'ordinary 13': old,
                'swapped y': frames,
            }))
    json_to(out / 'results.json', dict(previous=PARENT, script_sha=digest(Path(__file__)),
            protocol_sha=digest(Path(__file__).with_name('mid-wall-channel-replay-protocol.md')), cases=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
