"""Compute a weighted L1 lower bound for position-independent RGB transforms."""
import argparse
import json
from pathlib import Path

import numpy as np

from background_intervention import edit_weight, target_regions
from independent_coco import digest
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import mask_for
from probe_neutral_relation_balance import checked_image
from probe_reference_chroma_spread import BASE, PARENT, ROOT


def rgb_keys(pixels):
    return pixels.astype(np.int32) @ np.array([65536, 256, 1], dtype=np.int32)


def weighted_optimum(codes, desired, weights):
    unique, groups = np.unique(codes, return_inverse=True)
    result = np.empty_like(desired)
    for channel, values in enumerate(desired.T):
        order = np.lexsort((values, groups))
        starts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1]
        sorted_weights = weights[order]
        half = np.add.reduceat(sorted_weights, starts) / 2
        end = np.cumsum(sorted_weights)[np.r_[starts[1:] - 1, len(order) - 1]]
        offsets = np.searchsorted(np.cumsum(sorted_weights), end - half)
        assert len(offsets) == len(unique)
        result[:, channel] = values[order[offsets]][groups]
    return result


def run(out):
    reference = ROOT / 'docs/research/2026-11-18-local-edit-toggle-evidence.json'
    assert digest(reference) == PARENT_REPORT
    prior = json.loads(reference.read_bytes())
    parent = PARENT / 'results.json'
    assert digest(parent) == prior['parent_sha256']
    manifest = {row['pair']: row for row in json.loads(parent.read_bytes())['cases']}
    cases = {}
    for name, record in prior['cases'].items():
        folder = BASE / 'local-edit-toggle-v1' / name
        source, base, local = [checked_image(folder / file, record['files'][file])
                               for file in ('target.png', 'base-expected.png', 'local-expected.png')]
        human = mask_for(manifest[name], name.split('-')[0])
        if not human.any():
            cases[name] = dict(identity=bool(np.array_equal(base, local)))
            continue
        core = edit_weight(human, 1)[0] == 1
        neutral = human & target_regions(source, human)['neutral_proxy']
        assert not (core & neutral).any() and core.sum() > 256 and neutral.sum() >= 64
        selection = core | neutral
        codes = rgb_keys(source[selection])
        core_rows = core[selection]
        desired = (local.astype(np.int16) - base)[selection]
        # Integer weights preserve the ratio 1/(3*n_core) to 1/(1.5*n_neutral).
        weights = np.where(core_rows, int(neutral.sum()), 2 * int(core.sum()))
        output = weighted_optimum(codes, desired, weights)
        error = np.abs(output - desired)
        foreground, background = float(error[~core_rows].mean()), float(error[core_rows].mean())
        shared = np.intersect1d(codes[core_rows], codes[~core_rows])
        joint = foreground / 1.5 + background / 3
        cases[name] = dict(core_pixels=int(core.sum()), neutral_person_pixels=int(neutral.sum()),
                           shared_colors=len(shared), core_shared_pixels=int(np.isin(codes[core_rows], shared).sum()),
                           neutral_shared_pixels=int(np.isin(codes[~core_rows], shared).sum()),
                           core_mae=background, neutral_person_mae=foreground, scaled_lower_bound=joint,
                           excludes_joint_pass=joint > 2)
    source = Path(__file__)
    report = dict(cases=cases, toggle_sha256=PARENT_REPORT, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('global-local-capacity-protocol.md')))
    out.mkdir()
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
