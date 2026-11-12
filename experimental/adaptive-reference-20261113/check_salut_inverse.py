"""Fixed Standard inverse -> SA-LUT, on previously used development pairs only."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from salut_model import ROOT, SALUTModel
from salut_domain import fit_inverse, apply_inverse, quantize_log_input
from check_salut import metrics


@torch.inference_mode()
def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/pst50'
    # Preserve the first neutral-initialization run as a failed solver diagnostic.
    out = data / 'development-salut-inverse-v2'
    out.mkdir(exist_ok=True)
    model = SALUTModel()
    started = time.perf_counter()
    inverse, fit = fit_inverse(model.model.standard_lut, dim=33, steps=20)
    torch.cuda.synchronize()
    fit['seconds'] = time.perf_counter()-started
    print(json.dumps({'fixed_table_fit': fit}), flush=True)
    np.savez_compressed(out / 'fixed-inverse.npz', inverse=inverse.cpu().numpy())
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    expected = {item['path']: item['sha256'] for item in manifest['images']}
    old = json.loads((data / 'development-salut/results.json').read_text(encoding='utf-8'))
    native = json.loads((data / 'development-salut-native-log/results.json').read_text(encoding='utf-8'))
    rows = []
    for index in range(1, 6):
        for relative in (f'content_709/in{index}.png', f'paired_style/tar{index}.png'):
            if hashlib.sha256((data / relative).read_bytes()).hexdigest() != expected[relative]:
                raise ValueError('Development input changed')
        target = np.array(Image.open(data / f'content_709/in{index}.png').convert('RGB'))
        reference = np.array(Image.open(data / f'paired_style/tar{index}.png').convert('RGB'))
        start = time.perf_counter()
        original_rgb = model._tensor(target)
        log_tensor, prequant_error = apply_inverse(inverse, model.model.standard_lut, original_rgb)
        codes, roundtrip_error = quantize_log_input(model.model.standard_lut, log_tensor, original_rgb)
        constructed = codes[0].permute(1, 2, 0).cpu().numpy()
        image_error = roundtrip_error.amax(dim=1)
        domain = {'conversion_seconds': time.perf_counter()-start,
                  'prequantization_roundtrip_rgb_mae_codes': float(prequant_error.mean()*255),
                  'roundtrip_rgb_mae_codes': float(roundtrip_error.mean()*255),
                  'p95_max_channel_roundtrip_error_codes': float(torch.quantile(image_error.flatten(), .95)*255),
                  'fraction_pixels_over_2_codes': float((image_error > 2/255).float().mean())}
        result, diagnostic = model.predict(constructed, reference)
        # Paired target and real Log are diagnostic outputs only, never inversion inputs.
        truth_relative = f'paired_gt/gt{index}.png'
        if hashlib.sha256((data / truth_relative).read_bytes()).hexdigest() != expected[truth_relative]:
            raise ValueError('Development paired output changed')
        truth = np.array(Image.open(data / truth_relative).convert('RGB'))
        real_log = np.array(Image.open(data / f'content_log/in{index}.png').convert('RGB'))
        row = {'index': index, 'constructed_log_vs_real_log_rgb_mae': float(np.abs(constructed.astype(float)-real_log).mean()),
               'conversion': domain, 'diagnostics': diagnostic, 'inverse_salut': metrics(result, truth),
               'current': old['cases'][index-1]['current'], 'canonical': old['cases'][index-1]['canonical'],
               'salut_rec709': old['cases'][index-1]['salut'], 'salut_real_log': native['cases'][index-1]['salut']}
        rows.append(row)
        Image.fromarray(result).save(out / f'{index}-inverse-salut.png')
        Image.fromarray(constructed).save(out / f'{index}-constructed-log.png')
        current = np.array(Image.open(data / f'development-salut/{index}-Current.png').convert('RGB'))
        panel = Image.new('RGB', (360*3, 260*2), '#222222')
        draw = ImageDraw.Draw(panel)
        images = [('Input RGB', target), ('Reference', reference), ('Paired GT', truth),
                  ('Current', current), ('Constructed Log', constructed), ('Inverse + SA-LUT', result)]
        for n, (name, rgb) in enumerate(images):
            im = Image.fromarray(rgb)
            im.thumbnail((356, 232))
            x, y = n%3*360, n//3*260
            panel.paste(im, (x+(360-im.width)//2, y+24+(232-im.height)//2))
            draw.text((x+8, y+6), name, fill='white')
        panel.save(out / f'{index}-comparison.jpg', quality=94)
        report = {'scope': 'First five development pairs; fixed LUT-only inverse, not native Log recovery or product integration',
                  'fit': fit, 'protocol_sha256': hashlib.sha256(Path(__file__).with_name('salut-domain-protocol.md').read_bytes()).hexdigest(),
                  'inverse_source_sha256': hashlib.sha256(Path(__file__).with_name('salut_domain.py').read_bytes()).hexdigest(),
                  'baseline_report_sha256': hashlib.sha256((data / 'development-salut/results.json').read_bytes()).hexdigest(),
                  'native_log_report_sha256': hashlib.sha256((data / 'development-salut-native-log/results.json').read_bytes()).hexdigest(),
                  'cases': rows}
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
