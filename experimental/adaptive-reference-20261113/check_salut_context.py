"""Frozen development ablation of spatial context, not an output selector."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from salut_model import ROOT, SALUTModel
from salut_interpolation import quadrilinear
from check_salut import metrics


@torch.inference_mode()
def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/pst50'
    out = data / 'development-context-ablation'
    out.mkdir(exist_ok=True)
    model = SALUTModel()
    rows = []
    for index in range(1, 6):
        target = np.array(Image.open(data / f'content_log/in{index}.png').convert('RGB'))
        reference = np.array(Image.open(data / f'paired_style/tar{index}.png').convert('RGB'))
        full, diagnostic = model.predict(target, reference)
        lut, context = model.last_lut[0], model.last_context
        resized = F.interpolate(context, target.shape[:2], mode='bilinear', align_corners=True)
        original = model._tensor(target)
        controls = {'spatial': full}
        for name, scalar in [('mean', float(resized.mean())), ('zero', 0.), ('one', 1.)]:
            fixed = torch.full_like(resized, scalar)
            output = quadrilinear(lut, torch.cat((fixed, original), dim=1))
            controls[name] = output[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
        truth = np.array(Image.open(data / f'paired_gt/gt{index}.png').convert('RGB'))
        if truth.shape != target.shape:
            raise ValueError('Unaligned ground truth')
        scores = {name: metrics(rgb, truth) for name, rgb in controls.items()}
        mean_difference = float(np.abs(full.astype(float)-controls['mean']).mean())
        row = {'index': index, 'scores': scores, 'spatial_vs_mean_rgb_mae': mean_difference,
               'diagnostics': diagnostic, 'context_shape': list(context.shape)}
        rows.append(row)
        context_image = (resized[0, 0].cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(context_image).save(out / f'{index}-context.png')
        panel = Image.new('RGB', (360*3, 260*2), '#222222')
        draw = ImageDraw.Draw(panel)
        for n, (name, rgb) in enumerate([('Paired GT', truth), *controls.items(), ('Context', context_image)]):
            im = Image.fromarray(rgb).convert('RGB')
            im.thumbnail((356, 232))
            x, y = n%3*360, n//3*260
            panel.paste(im, (x+(360-im.width)//2, y+24+(232-im.height)//2))
            draw.text((x+8, y+6), name, fill='white')
        panel.save(out / f'{index}-ablation.jpg', quality=94)
        print(json.dumps(row), flush=True)
        report = {'scope': 'Development-only Log inputs; fixed LUT, spatial vs constant context',
                  'protocol_sha256': hashlib.sha256(Path(__file__).with_name('salut-context-protocol.md').read_bytes()).hexdigest(),
                  'cases': rows}
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
