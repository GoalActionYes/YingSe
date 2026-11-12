"""Characterize input-domain conversion on the SAME five development originals.

No model tuning. Compare author Standard.cube(Log) with dataset-provided Rec709.
This does not establish an inverse or recover camera data from arbitrary photos.
"""
import json
import numpy as np
from PIL import Image
import torch
from salut_model import ROOT, SALUTModel
from salut_interpolation import trilinear


@torch.inference_mode()
def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/pst50'
    out = data / 'development-domain-probe'
    out.mkdir(exist_ok=True)
    model = SALUTModel()
    rows = []
    for index in range(1, 6):
        log = np.array(Image.open(data / f'content_log/in{index}.png').convert('RGB'))
        rec709 = np.array(Image.open(data / f'content_709/in{index}.png').convert('RGB'))
        converted = trilinear(model.model.standard_lut, model._tensor(log))[0].permute(1, 2, 0).clamp(0, 1)
        result = converted.mul(255).round().byte().cpu().numpy()
        if result.shape != rec709.shape:
            raise ValueError('Mismatched development inputs')
        error = np.abs(result.astype(float)-rec709.astype(float))
        Image.fromarray(result).save(out / f'{index}-standard-converted.png')
        row = {'index': index, 'rgb_mae': float(error.mean()), 'p95': float(np.percentile(error, 95)),
               'max_error': float(error.max()), 'clipped_channel_fraction': float(((result == 0) | (result == 255)).mean())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'scope': 'Five development Log/Rec709 originals, fixed author Standard.cube, no inverse fitted', 'cases': rows}
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
