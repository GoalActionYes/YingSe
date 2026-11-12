"""Ordinary-RGB development controls for inverse-adapted SA-LUT, no oracle input."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
from salut_model import ROOT, SALUTModel
from salut_domain import fit_inverse, apply_inverse, quantize_log_input
from check_semantic_crops import score


@torch.inference_mode()
def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/independent-coco'
    jobs_path = data / 'development/jobs.json'
    jobs = json.loads(jobs_path.read_text(encoding='utf-8'))
    if jobs['split'] != 'development':
        raise ValueError('Frozen evaluation data must remain untouched')
    baseline_path = data / 'development/semantic-candidate/results.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    prior = {(r['target'], r['reference'], r['profile']): r for r in baseline['cases']}
    out = data / 'development/salut-inverse'
    out.mkdir(exist_ok=True)
    model = SALUTModel()
    inverse, fit = fit_inverse(model.model.standard_lut, dim=33, steps=20)
    rows = []
    for job in jobs['jobs']:
        key = job['target'], job['reference'], job['profile']
        previous = prior[key]
        target = np.array(Image.open(data / f'{job["target"]}-input.png').convert('RGB'))
        reference = np.array(Image.open(data / job['reference_file']).convert('RGB'))
        original_rgb = model._tensor(target)
        converted, prequant_error = apply_inverse(inverse, model.model.standard_lut, original_rgb)
        codes, error = quantize_log_input(model.model.standard_lut, converted, original_rgb)
        constructed = codes[0].permute(1, 2, 0).cpu().numpy()
        output, diagnostics = model.predict(constructed, reference)
        # Human masks and known transformed target are read AFTER model prediction.
        truth = np.array(Image.open(data / job['truth_file']).convert('RGB'))
        mask = np.asarray(Image.open(data / f'{job["target"]}-human-mask.png').convert('L').filter(ImageFilter.GaussianBlur(2)), dtype=float)/255
        row = {'target': key[0], 'reference': key[1], 'profile': key[2], 'group': job['group'],
               'baseline': previous['baseline'], 'canonical': previous['model'], 'salut_inverse': score(output, truth, mask),
               'diagnostics': diagnostics, 'conversion_mae_codes': float(error.mean()*255),
               'prequantization_conversion_mae_codes': float(prequant_error.mean()*255),
               'conversion_pixels_over_2_codes': float((error.amax(dim=1) > 2/255).float().mean())}
        Image.fromarray(output).save(out / f'{key[0]}-{key[2]}.png')
        rows.append(row)
        report = {'scope': 'Six previously inspected COCO development pairs, synthetic known styles; no heldout or aesthetic claim',
                  'fixed_inverse_fit': fit, 'baseline_report_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  'jobs_sha256': hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
                  'inverse_source_sha256': hashlib.sha256(Path(__file__).with_name('salut_domain.py').read_bytes()).hexdigest(), 'cases': rows}
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'target': key[0], 'profile': key[2], 'current': row['baseline']['mae'],
                          'canonical': row['canonical']['mae'], 'salut_inverse': row['salut_inverse']['mae']}), flush=True)


if __name__ == '__main__':
    main()
