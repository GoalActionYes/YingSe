"""Hold foreground harmonization fixed and substitute a known person mask offline."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch

from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import ROLES, mask_for
from probe_neutral_relation_balance import controlled as local_controls
from probe_person_harmonizer import PROFILES, Probe, WEIGHT_SHA
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision, read
from probe_roundtrip_grade_choice import checked_json, read_rgb


PREVIOUS = BASE / 'person-harmonizer-v1'
PREVIOUS_SHA = 'cc714135d728615179d1a629134295c7c4156c69f6ed8f3fb0e5074f901485d2'


class KnownMaskProbe(Probe):
    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def probability(self, shape, data):
        return self.masks[(shape, data)]


def edges(mask):
    bitmap = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(bitmap.filter(ImageFilter.MaxFilter(7))) > 0
    eroded = np.asarray(bitmap.filter(ImageFilter.MinFilter(7))) > 0
    return {'inner_edge': mask & ~eroded, 'outer_edge': dilated & ~mask,
            'person_core': eroded, 'background_core': ~dilated,
            'whole': np.ones_like(mask)}


def difference(new, old, scopes):
    delta = np.abs(new.astype(np.int16) - old)
    return {name: float(delta[mask].mean()) if mask.any() else None for name, mask in scopes.items()}


def run(out):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(ROOT / 'docs/research/2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    past = checked_json(PREVIOUS / 'results.json', PREVIOUS_SHA)
    assert toggle['parent_sha256'] == PARENT_SHA and manifest['strength'] == .8
    assert past['weight_sha256'] == WEIGHT_SHA and past['background_sha256'] == PARENT_SHA
    parent = {row['pair']: row for row in manifest['cases']}
    targets = {pair: read(row['input_file']) for pair, row in parent.items()}
    masks = {pair: mask_for(row, pair.split('-')[0]) for pair, row in parent.items()}
    frozen = {(targets[pair].shape, targets[pair].tobytes()): mask.astype(np.float32) for pair, mask in masks.items()}
    if len(frozen) != 6:
        raise ValueError('Target photographs unexpectedly repeated')
    out.mkdir(exist_ok=False)
    probe = KnownMaskProbe(frozen)
    mask_quality = {}
    for pair, target in targets.items():
        authored = masks[pair]
        automatic, _ = probe.segmenter.predict(target)
        labelled = automatic >= .5
        region = authored | labelled
        mask_quality[pair] = dict(iou=float((authored & labelled).sum() / region.sum()) if region.any() else None,
                                  pixels=int(authored.sum()), classified=int(labelled.sum()),
                                  soft_mae=float(np.abs(automatic - authored).mean()))
    arms = {}
    for profile in PROFILES:
        folder = out / profile
        folder.mkdir()
        probe.profile = profile
        sample = np.full((32, 32, 3), 127, np.uint8)
        np.testing.assert_array_equal(probe.measure(sample, sample.copy(), sample - 20)[0], sample)
        background = background_controls(folder, manifest, probe.measure)
        local = local_controls(probe, folder, toggle, parent)
        previous = past['arms'][profile]
        image_effect = {}
        for pair, truth in masks.items():
            scopes = edges(truth)
            variants = {}
            for role in ROLES:
                filename = f'{pair}-{role}.png'
                baseline = read_rgb(PREVIOUS / profile / filename, previous['local'][pair]['output'][role]['sha256'])
                result = read_rgb(folder / filename, local[pair]['output'][role]['sha256'])
                variants[role] = difference(result, baseline, scopes)
            image_effect[pair] = variants
        arms[profile] = dict(background=background, background_gate=decision(background), local=local,
                             mask_effect=image_effect)
    seconds = time.monotonic() - started
    if seconds > 100:
        raise TimeoutError('Frozen mask check budget exceeded')
    script = Path(__file__)
    result = dict(arms=arms, mask_quality=mask_quality, same_image=True, seconds=seconds,
                  source_sha256=digest(script), old_result_sha256=PREVIOUS_SHA, parent_sha256=PARENT_SHA,
                  protocol_sha256=digest(script.with_name('person-harmonizer-mask-check-protocol.md')),
                  weight_sha256=WEIGHT_SHA, toggle_sha256=PARENT_REPORT)
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: arm['background_gate'] for key, arm in arms.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
