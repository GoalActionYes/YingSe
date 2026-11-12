"""Recheck five runtime failures after a bounded-interpolation roundoff fix.

Not a new blind benchmark. Original 120-case report is never modified. Require
identical assets, thresholds and source except the audited interpolation helper.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
from salut_model import ROOT, SALUTModel
from salut_domain import fit_inverse, apply_inverse, quantize_log_input
from canonical_model import CanonicalModel
from check_semantic_crops import score
from salut_holdout import assess, summarize, validate_jobs
from reference_color_app.pipeline import analyze, algorithm_fingerprint
from reference_color_app.lut import apply_mapping


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unchanged_except_sample(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == '_sample_inverse')]
    return ast.dump(tree, include_attributes=False)


@torch.inference_mode()
def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/independent-coco'
    first = data / 'evaluation/salut-inverse-first-holdout'
    original = json.loads((first / 'results.json').read_text(encoding='utf-8'))
    lock = json.loads((first / 'run-lock.json').read_text(encoding='utf-8'))
    if len(original['cases']) != 120 or original['run_lock_sha256'] != sha(first / 'run-lock.json'):
        raise ValueError('Original first holdout is missing or changed')
    source = Path(__file__).parent
    for name, expected in lock['research_sha256'].items():
        old_path = first / 'frozen-source' / name
        if sha(old_path) != expected:
            raise ValueError('Frozen source archive changed')
        if name == 'salut_domain.py':
            if unchanged_except_sample(old_path.read_text(encoding='utf-8')) != unchanged_except_sample((source / name).read_text(encoding='utf-8')):
                raise ValueError('Solver change exceeds audited interpolation helper')
        elif sha(source / name) != expected:
            raise ValueError('Frozen algorithm/protocol changed: '+name)
    if algorithm_fingerprint() != lock['algorithm_fingerprint']:
        raise ValueError('Original algorithm changed')
    for relative, expected in lock['asset_sha256'].items():
        if sha(data / relative) != expected:
            raise ValueError('Heldout asset changed: '+relative)
    jobs = json.loads((data / 'evaluation/jobs.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    validate_jobs(jobs, manifest)
    job_map = {(j['target'], j['reference'], j['profile']): j for j in jobs['jobs']}
    failed = [r for r in original['cases'] if 'error' in r]
    if len(failed) != 5 or any(r['error'] != 'ValueError: Interpolation input must be finite and within [0,1]' for r in failed):
        raise ValueError('Unexpected failures; this repair is limited to the five roundoff cases')
    out = data / 'evaluation/salut-inverse-roundoff-repair'
    out.mkdir(exist_ok=False)
    repair_lock = {'scope': 'Post-holdout numerical repair, not a fresh blind evaluation',
                   'original_results_sha256': sha(first / 'results.json'), 'original_lock_sha256': sha(first / 'run-lock.json'),
                   'repaired_solver_sha256': sha(source / 'salut_domain.py'), 'repair_script_sha256': sha(Path(__file__)),
                   'changed_function_only': '_sample_inverse', 'unchanged_case_count': 115}
    (out / 'repair-lock.json').write_text(json.dumps(repair_lock, indent=2), encoding='utf-8')
    model = SALUTModel()
    canonical = CanonicalModel()
    inverse, fit = fit_inverse(model.model.standard_lut, dim=33, steps=20)
    replacements = {}
    for old in failed:
        key = old['target'], old['reference'], old['profile']
        job = job_map[key]
        target_path, reference_path = data / f'{key[0]}-input.png', data / job['reference_file']
        target = np.array(Image.open(target_path).convert('RGB'))
        reference = np.array(Image.open(reference_path).convert('RGB'))
        baseline = apply_mapping(target, analyze(target_path, reference_path), 1.)
        global_rgb, _ = canonical.predict(target, reference)
        rgb_tensor = model._tensor(target)
        continuous, _ = apply_inverse(inverse, model.model.standard_lut, rgb_tensor)
        codes, error = quantize_log_input(model.model.standard_lut, continuous, rgb_tensor)
        candidate, diagnostic = model.predict(codes[0].permute(1, 2, 0).cpu().numpy(), reference)
        truth = np.array(Image.open(data / job['truth_file']).convert('RGB'))
        mask = np.asarray(Image.open(data / f'{key[0]}-human-mask.png').convert('L').filter(ImageFilter.GaussianBlur(2)), dtype=float)/255
        row = {k: job[k] for k in ('target', 'reference', 'profile', 'group')}
        row.update(baseline=score(baseline, truth, mask), canonical=score(global_rgb, truth, mask),
                   salut_inverse=score(candidate, truth, mask), diagnostics=diagnostic,
                   conversion_mae_codes=float(error.mean()*255),
                   conversion_pixels_over_2_codes=float((error.amax(dim=1) > 2/255).float().mean()))
        replacements[key] = row
        for label, image in [('baseline', baseline), ('canonical', global_rgb), ('salut', candidate)]:
            Image.fromarray(image).save(out / f'{key[0]}-{key[2]}-{label}.png')
        print(json.dumps({'repaired_target': key[0], 'profile': key[2], 'mae': row['salut_inverse']['mae']}), flush=True)
    merged = [replacements[(r['target'], r['reference'], r['profile'])] if 'error' in r else r for r in original['cases']]
    report = {'scope': 'Five post-holdout roundoff repairs merged with 115 original rows unchanged; not a new blind benchmark',
              'repair_lock_sha256': sha(out / 'repair-lock.json'), 'fixed_inverse_fit': fit,
              'gate': assess(merged), 'summary': summarize(merged),
              'by_profile': {p: summarize([r for r in merged if r['profile'] == p]) for p in sorted({r['profile'] for r in merged})},
              'by_group': {g: summarize([r for r in merged if r['group'] == g]) for g in sorted({r['group'] for r in merged})},
              'cases': merged}
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'repaired_gate': report['gate']}), flush=True)


if __name__ == '__main__':
    main()
