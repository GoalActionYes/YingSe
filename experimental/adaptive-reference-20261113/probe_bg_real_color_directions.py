"""Screen a fixed background-swap color projection on held-out development controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from fetch_bg_real_color_directions import DEV_SHA, DOCS, read_pinned, select
from independent_coco import digest
from probe_bg_real_pixel_pilot import interior, measure
from probe_gray_edge_correction import PARENT_SHA, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import mask_for, save
from probe_neutral_relation_balance import checked_image, controlled as local_controls
from probe_reference_chroma_spread import PARENT, decision, read
from reference_color_app.media import decode_image
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


def cdf(histogram):
    return np.cumsum(histogram[0, 1:].cpu().numpy(), axis=-1, dtype=np.float64)


def learn_basis(probe, inputs, manifest, selection):
    rows, deltas = {}, []
    for key, item in selection:
        files = manifest[key]
        pictures = {}
        for role, row in files.items():
            path = inputs / row['name']
            assert path.parent == inputs and digest(path) == row['sha256']
            if role == 'mask':
                assert row['path'] == item['subject_mask']
                with Image.open(path) as label:
                    pictures[role] = np.asarray(label.convert('L')) >= 128
            else:
                assert row['sha256'] == item['arms'][role]['sha256']
                pictures[role] = decode_image(path)
        first, second, mask = (pictures[role] for role in ('C1', 'F1', 'mask'))
        height, width = mask.shape
        assert item['shape'] == [width, height] and first.shape == second.shape == (height, width, 3)
        core, outside = measure(first, second, interior(mask)), measure(first, second, interior(~mask))
        qualified = core['pixels'] >= 256 and outside['pixels'] >= 256 and core['mae'] <= 1 and outside['mae'] >= 8
        rows[key] = dict(hashes={role: row['sha256'] for role, row in files.items()},
                         subject_core=core, background_core=outside, qualified=qualified)
        if qualified:
            delta = (cdf(probe.hist(second)) - cdf(probe.hist(first))).ravel()
            length = np.linalg.norm(delta)
            if length <= 1e-12:
                raise ValueError('Qualified image pair has no measurable chroma CDF difference')
            deltas.append(delta / length)
    if len(deltas) < 8:
        raise ValueError('Fewer than eight usable fitting pairs')
    _, singular, directions = np.linalg.svd(np.stack(deltas), full_matrices=False)
    return directions[:2], dict(rows=rows, qualified=len(deltas), singular_values=singular.tolist())


def adjusted_histogram(target_hist, reference_hist, basis):
    source, reference = cdf(target_hist), cdf(reference_hist)
    difference = (reference - source).ravel()
    removed = (difference @ basis.T) @ basis
    projected = reference - removed.reshape(reference.shape)
    monotone = np.maximum.accumulate(projected.clip(0, 1), axis=1)
    monotone[:, -1] = 1
    masses = np.diff(monotone, axis=1, prepend=0)
    np.testing.assert_allclose(masses.sum(axis=1), 1, atol=1e-12, rtol=0)
    assert (masses >= 0).all()
    result = reference_hist.clone()
    result[0, 1:] = torch.as_tensor(masses, device=result.device, dtype=result.dtype)
    return result, dict(removed_norm=float(np.linalg.norm(removed)),
                        repaired_norm=float(np.linalg.norm(monotone - projected)))


class ProjectionProbe:
    def __init__(self, engine, basis):
        self.engine = engine
        self.basis = basis

    @torch.inference_mode()
    def hist(self, rgb):
        api = self.engine.api
        return api['batch_histogram_1D'](api['srgb_to_ycbcr'](self.engine.analysis(rgb)))

    @torch.inference_mode()
    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(same_image=True)
        engine, api = self.engine, self.engine.api
        target_hist = self.hist(target)
        reference_hist, debug = adjusted_histogram(target_hist, self.hist(reference), self.basis)
        low = api['TF'].resize(engine.analysis(target), (64, 64), antialias=True)
        grid = api['lowpass'](engine.model.apply_grid(low, torch.cat((target_hist, reference_hist), dim=1)))
        assert tuple(grid.shape) == (1, 13, 8, 16, 16) and torch.isfinite(grid).all()
        return finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8), debug


class FrozenBaseline:
    @staticmethod
    def measure(target, reference, baseline):
        return baseline, {}


def photo_controls(output, held_inputs, held_outputs, held, parent, probe):
    records = {}
    for key, row in held['cases'].items():
        target_pair = row['target_pair']
        entry = parent[target_pair]
        target = read(entry['input_file'])
        outputs = {}
        hashes = {}
        baseline = {}
        for role in ('C1', 'F1'):
            member = row['input'][role]
            path = held_inputs / member['name']
            assert digest(path) == member['sha256']
            reference = decode_image(path)
            baseline[role] = checked_image(held_outputs / f'{key}-{role}.png', row['outputs'][role])
            if key == next(iter(held['cases'])):
                frozen = finish_spatial(target, probe.engine.prepare(target, reference).apply(target), .8)
                np.testing.assert_array_equal(frozen, baseline[role])
            outputs[role], info = probe.measure(target, reference, baseline[role])
            hashes[role] = dict(info, sha256=save(output / f'{key}-{role}.png', outputs[role]))
        person = mask_for(entry, target_pair.split('-')[0])
        regions = target_regions(target, person)
        regions['neutral_person'] = person & regions['neutral_proxy']
        scores = {name: dict(baseline=measure(baseline['C1'], baseline['F1'], area),
                             candidate=measure(outputs['C1'], outputs['F1'], area))
                  for name, area in regions.items()}
        a, b = (scores['neutral_person'][label]['mae'] for label in ('baseline', 'candidate'))
        records[key] = dict(target_pair=target_pair, scores=scores, output=hashes,
                            safe=b <= a + 1, passed=b <= .85 * a)
    return records


def local_summary(cases):
    return {key: dict(gates=row['gates'], local_background_projection=(
        row['metrics']['local']['edited_core'] or {}).get('projection'),
        whole_projection=row['metrics']['whole']['whole']['projection'],
        neutral_person_error=(row['metrics']['local']['neutral_person'] or {}).get('error'),
        output=row['output']) for key, row in cases.items()}


def local_positive_pass(row):
    return row['gates']['positive'] and all(value is not False for key, value in row['gates']['checks'].items()
                                            if key.startswith('local_'))


def run(inputs, output):
    started = time.monotonic()
    torch.set_num_threads(4)
    _, held, selection = select()
    manifest_path = inputs / 'intake.json'
    manifest = json.loads(manifest_path.read_bytes())
    assert set(manifest) == {key for key, _ in selection}
    train_hashes = {m['sha256'] for row in manifest.values() for m in row.values()}
    held_hashes = {m['sha256'] for row in held['cases'].values() for m in row['input'].values()}
    assert len(train_hashes) == 36 and not train_hashes & held_hashes
    parent = read_pinned(PARENT / 'results.json', PARENT_SHA)
    toggle = read_pinned(DOCS / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == held['checkpoint_sha256'] == parent['model_sha256']
    probe = ProjectionProbe(engine, None)
    basis, fit = learn_basis(probe, inputs, manifest, selection)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), rtol=0, atol=1e-12)
    probe.basis = basis

    image = np.full((32, 32, 3), 123, dtype=np.uint8)
    np.testing.assert_array_equal(probe.measure(image, image.copy(), 255 - image)[0], image)
    output.mkdir(exist_ok=False)
    source = Path(__file__)
    parent_index = {case['pair']: case for case in parent['cases']}
    photo = photo_controls(output, inputs.parent / 'bg-real-pixel-inputs', inputs.parent / 'bg-real-pixel-pilot',
                           held, parent_index, probe)
    background = background_controls(output, parent, probe.measure)
    local = local_summary(local_controls(probe, output, toggle, parent_index))
    baseline_dir = output / 'local_baseline'
    baseline_dir.mkdir()
    local_base = local_summary(local_controls(FrozenBaseline(), baseline_dir, toggle, parent_index))
    np.save(output / 'basis.npy', basis, allow_pickle=False)

    bg = decision(background)
    local_checks = dict(
        count=sum(local_positive_pass(row) for row in local.values()) >= sum(
            local_positive_pass(row) for row in local_base.values()),
        background_direction=all(row['local_background_projection'] >= local_base[key]['local_background_projection'] - .10
                                 for key, row in local.items() if row['gates']['positive']),
        whole_response=sum(row['gates']['checks']['whole_response'] for row in local.values()) >= sum(
            row['gates']['checks']['whole_response'] for row in local_base.values()),
        without_person=all(row['gates']['checks']['no_person'] is not False for row in local.values()))
    photo_pass = sum(row['passed'] for row in photo.values())
    gates = dict(photo=photo_pass >= 3 and all(row['safe'] for row in photo.values()),
                 existing_background=bg['proceed'], local=all(local_checks.values()))
    assert time.monotonic() - started <= 120
    report = dict(held_sha256=DEV_SHA, parent_sha256=PARENT_SHA, local_sha256=PARENT_REPORT,
                  fit_manifest_sha256=digest(manifest_path), weight_sha256=probe.engine.checkpoint_sha,
                  basis_sha256=digest(output / 'basis.npy'), script_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('bg-real-color-directions-protocol.md')),
                  fit=fit, photo=photo, photo_pass=photo_pass, background=background, background_gate=bg,
                  local=local, local_base=local_base, local_checks=local_checks, gates=gates,
                  proceed=all(gates.values()))
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(qualified=fit['qualified'], photo_pass=photo_pass, background=bg,
                          local_checks=local_checks, gates=gates), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.inputs, args.output)
