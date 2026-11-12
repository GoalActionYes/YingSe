"""Frozen first holdout run. No per-example tuning or algorithm selection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_jobs(jobs, manifest):
    profiles = ('identity', 'global_contrast', 'global_desaturate', 'region_color_luminance', 'region_tone_saturation')
    pairs = [p for p in manifest['pairs'] if p['split'] == 'evaluation']
    if len(pairs) != 24 or Counter(p['group'] for p in pairs) != {'single': 8, 'multiple': 8, 'none': 8}:
        raise ValueError('Original heldout membership must contain 24 prescribed pairs')
    expected = {(p['target'], p['reference'], p['group'], profile) for p in pairs for profile in profiles}
    if jobs['split'] != 'evaluation' or len(jobs['jobs']) != 120:
        raise ValueError('Expected original 24 heldout pairs and five profiles')
    observed = {(j['target'], j['reference'], j['group'], j['profile']) for j in jobs['jobs']}
    if observed != expected:
        raise ValueError('Job membership differs from original evaluation pairs')
    for j in jobs['jobs']:
        if (j['split'] != 'evaluation'
                or j['reference_file'].replace('\\', '/') != f'evaluation/references/{j["reference"]}-{j["profile"]}.png'
                or j['truth_file'].replace('\\', '/') != f'evaluation/truth/{j["target"]}-{j["profile"]}.png'):
            raise ValueError('Unexpected heldout asset binding')


def assess(rows):
    valid = [r for r in rows if 'error' not in r]
    keys = [(r['target'], r['reference'], r['profile']) for r in valid]
    groups = Counter(r['group'] for r in valid)
    profiles = Counter(r['profile'] for r in valid)
    expected_profiles = {'identity', 'global_contrast', 'global_desaturate', 'region_color_luminance', 'region_tone_saturation'}
    complete = (len(rows) == 120 and len(valid) == 120 and len(set(keys)) == 120
                and groups == {'single': 40, 'multiple': 40, 'none': 40}
                and set(profiles) == expected_profiles and all(v == 24 for v in profiles.values()))
    regional = [r for r in valid if r['group'] != 'none' and r['profile'].startswith('region_')]
    result = {'count': len(rows), 'valid': len(valid), 'regional_count': len(regional), 'complete': complete, 'passed': False}
    if not valid or not regional:
        return result
    for r in valid:
        if not all(np.isfinite(r[name]['mae']) for name in ('baseline', 'salut_inverse')):
            return result
    for r in regional:
        if not all(r[name]['relation_error'] is not None and np.isfinite(r[name]['relation_error']) for name in ('baseline', 'salut_inverse')):
            return result
    base = np.array([r['baseline']['mae'] for r in valid])
    candidate = np.array([r['salut_inverse']['mae'] for r in valid])
    rb = np.array([r['baseline']['relation_error'] for r in regional])
    rc = np.array([r['salut_inverse']['relation_error'] for r in regional])
    severe = int(((candidate-base > 5) & (candidate > base*1.25)).sum())
    result.update(median_baseline=float(np.median(base)), median_candidate=float(np.median(candidate)),
                  improved_count=int((base-candidate > .1).sum()), severe_regressions=severe,
                  median_regional_baseline=float(np.median(rb)), median_regional_candidate=float(np.median(rc)),
                  regional_improved_count=int((rb-rc > .1).sum()))
    gates = {'whole_median_10_percent': np.median(candidate) <= .9*np.median(base),
             'whole_improvement_65_percent': (base-candidate > .1).mean() >= .65,
             'regional_median_10_percent': np.median(rc) <= .9*np.median(rb),
             'regional_improvement_60_percent': (rb-rc > .1).mean() >= .6,
             'severe_regressions_at_most_10_percent': severe/len(valid) <= .1}
    result['gates'] = {k: bool(v) for k, v in gates.items()}
    result['passed'] = bool(complete and len(regional) == 32 and all(gates.values()))
    return result


def summarize(rows):
    valid = [r for r in rows if 'error' not in r]
    result = {'count': len(rows), 'valid': len(valid)}
    for method in ('baseline', 'canonical', 'salut_inverse'):
        relational = [r[method]['relation_error'] for r in valid if r[method]['relation_error'] is not None]
        result[method] = {'median_mae': float(np.median([r[method]['mae'] for r in valid])) if valid else None,
                          'median_relation_error': float(np.median(relational)) if relational else None}
    return result


def main():
    import torch
    from PIL import Image, ImageFilter, ImageDraw
    from salut_model import ROOT, SALUTModel, SOURCE_HASHES, CHECKPOINT_SHA
    from salut_domain import fit_inverse, apply_inverse, quantize_log_input
    from canonical_model import CanonicalModel, CHECKPOINT_SHA as CANON_SHA
    from check_semantic_crops import score
    from reference_color_app.pipeline import analyze, algorithm_fingerprint
    from reference_color_app.lut import apply_mapping
    data = ROOT / 'outputs/adaptive-reference-20261113/independent-coco'
    jobs_path = data / 'evaluation/jobs.json'
    jobs = json.loads(jobs_path.read_text(encoding='utf-8'))
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    validate_jobs(jobs, manifest)
    out = data / 'evaluation/salut-inverse-first-holdout'
    out.mkdir(exist_ok=False)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source_names = ('salut_model.py', 'salut_interpolation.py', 'salut_domain.py', 'canonical_model.py',
                    'controlled_styles.py', 'prepare_coco_controls.py', 'check_semantic_crops.py',
                    'salut_holdout.py', 'salut-holdout-protocol.md')
    asset_names = set()
    for job in jobs['jobs']:
        asset_names.update((job['reference_file'].replace('\\', '/'), job['truth_file'].replace('\\', '/')))
        for role in ('target', 'reference'):
            asset_names.update((f'{job[role]}-input.png', f'{job[role]}-human-mask.png'))
    # Hash oracle bytes for provenance without decoding or exposing them to inference.
    lock = {'scope': 'First evaluation; frozen 120 cases, no parameter update',
            'research_sha256': {name: sha(Path(__file__).with_name(name)) for name in source_names},
            'salut_author_source_sha256': SOURCE_HASHES, 'salut_checkpoint_sha256': CHECKPOINT_SHA,
            'canonical_checkpoint_sha256': CANON_SHA, 'algorithm_fingerprint': algorithm_fingerprint(),
            'jobs_sha256': sha(jobs_path), 'dataset_manifest_sha256': sha(data / 'manifest.json'),
            'asset_sha256': {name: sha(data / name) for name in sorted(asset_names)}}
    (out / 'run-lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    expected_inputs = {x['id']: x['input_sha256'] for x in manifest['images']}
    expected_masks = {x['id']: x['mask_sha256'] for x in manifest['images']}
    if jobs['controlled_style_source_sha256'] != sha(Path(__file__).with_name('controlled_styles.py')):
        raise ValueError('Control generation code changed')
    for job in jobs['jobs']:
        for role in ('target', 'reference'):
            if (sha(data / f'{job[role]}-input.png') != expected_inputs[job[role]]
                    or sha(data / f'{job[role]}-human-mask.png') != expected_masks[job[role]]):
                raise ValueError('Original input or mask identity changed')
    model = SALUTModel()
    canonical = CanonicalModel()
    inverse, fit = fit_inverse(model.model.standard_lut, dim=33, steps=20)
    rows = []
    with torch.inference_mode():
        for number, job in enumerate(jobs['jobs'], 1):
            name = f'{job["target"]}-{job["profile"]}'
            row = {k: job[k] for k in ('target', 'reference', 'profile', 'group')}
            try:
                target_path = data / f'{job["target"]}-input.png'
                reference_path = data / job['reference_file']
                target = np.array(Image.open(target_path).convert('RGB'))
                reference = np.array(Image.open(reference_path).convert('RGB'))
                baseline = apply_mapping(target, analyze(target_path, reference_path), 1.)
                global_rgb, _ = canonical.predict(target, reference)
                source_tensor = model._tensor(target)
                continuous, _ = apply_inverse(inverse, model.model.standard_lut, source_tensor)
                codes, error = quantize_log_input(model.model.standard_lut, continuous, source_tensor)
                constructed = codes[0].permute(1, 2, 0).cpu().numpy()
                candidate, diagnostic = model.predict(constructed, reference)
                # No oracle mask/desired output is read before all three predictions.
                truth = np.array(Image.open(data / job['truth_file']).convert('RGB'))
                mask = np.asarray(Image.open(data / f'{job["target"]}-human-mask.png').convert('L').filter(ImageFilter.GaussianBlur(2)), dtype=float)/255
                row.update(baseline=score(baseline, truth, mask), canonical=score(global_rgb, truth, mask),
                           salut_inverse=score(candidate, truth, mask), diagnostics=diagnostic,
                           conversion_mae_codes=float(error.mean()*255),
                           conversion_pixels_over_2_codes=float((error.amax(dim=1) > 2/255).float().mean()))
                for label, rgb in [('baseline', baseline), ('canonical', global_rgb), ('salut', candidate)]:
                    Image.fromarray(rgb).save(out / f'{name}-{label}.png')
                panel = Image.new('RGB', (360*3, 260*2), '#222222')
                draw = ImageDraw.Draw(panel)
                for n, (label, rgb) in enumerate([('Input', target), ('Reference', reference), ('Controlled target', truth),
                                                 ('Current', baseline), ('Canon E2E', global_rgb), ('SA-LUT inverse', candidate)]):
                    im = Image.fromarray(rgb)
                    im.thumbnail((356, 232))
                    x, y = n%3*360, n//3*260
                    panel.paste(im, (x+(360-im.width)//2, y+24+(232-im.height)//2))
                    draw.text((x+8, y+6), label, fill='white')
                panel.save(out / f'{name}-comparison.jpg', quality=92)
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            report = {'scope': 'First heldout synthetic-control evaluation; not a natural-aesthetic or video claim',
                      'run_lock_sha256': sha(out / 'run-lock.json'), 'fixed_inverse_fit': fit,
                      'gate': assess(rows), 'summary': summarize(rows),
                      'by_profile': {p: summarize([r for r in rows if r['profile'] == p]) for p in sorted({r['profile'] for r in rows})},
                      'by_group': {g: summarize([r for r in rows if r['group'] == g]) for g in sorted({r['group'] for r in rows})},
                      'cases': rows}
            (out / 'results.partial').write_text(json.dumps(report, indent=2), encoding='utf-8')
            (out / 'results.partial').replace(out / 'results.json')
            print(json.dumps({'case': number, 'total': 120, 'target': job['target'], 'profile': job['profile'],
                              'mae': row.get('salut_inverse', {}).get('mae'), 'error': row.get('error')}), flush=True)
    print(json.dumps({'final_gate': report['gate']}), flush=True)


if __name__ == '__main__':
    main()
