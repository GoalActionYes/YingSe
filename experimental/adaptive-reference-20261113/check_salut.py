"""Five development-only PST50 pairs; no frozen evaluation examples consumed."""
import hashlib
import json
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
from salut_model import ROOT, SALUTModel, SOURCE_HASHES, CHECKPOINT_SHA
from canonical_model import CanonicalModel
sys.path.insert(0, str(ROOT))
from reference_color_app.pipeline import analyze, algorithm_fingerprint
from reference_color_app.lut import apply_mapping


def metrics(output, truth):
    delta = output.astype(np.float64)-truth.astype(np.float64)
    mse = float(np.square(delta).mean())
    return {'rgb_mae': float(np.abs(delta).mean()), 'psnr': float(10*np.log10(255**2/mse)) if mse else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-log', action='store_true', help='Diagnostic only: real Log target, not synthesized from RGB')
    args = parser.parse_args()
    data = ROOT / 'outputs/adaptive-reference-20261113/pst50'
    out = data / ('development-salut-native-log' if args.native_log else 'development-salut')
    out.mkdir(exist_ok=True)
    model = SALUTModel()
    global_model = CanonicalModel()
    rows = []
    for index in range(1, 6):
        target_path = data / f'content_709/in{index}.png'
        reference_path = data / f'paired_style/tar{index}.png'
        target = np.array(Image.open(target_path).convert('RGB'))
        model_target = np.array(Image.open(data / f'content_log/in{index}.png').convert('RGB')) if args.native_log else target
        reference = np.array(Image.open(reference_path).convert('RGB'))
        print(f'Predicting development pair {index}, target={target.shape}, reference={reference.shape}', flush=True)
        candidate, diagnostic = model.predict(model_target, reference)
        diagnostic['input_domain'] = 'original Log' if args.native_log else 'Rec709 per upstream CLI example'
        print(json.dumps(diagnostic), flush=True)
        global_rgb, _ = global_model.predict(target, reference)
        baseline = apply_mapping(target, analyze(target_path, reference_path), 1.)
        # Only now read the paired expert output, after candidates are computed.
        truth = np.array(Image.open(data / f'paired_gt/gt{index}.png').convert('RGB'))
        if target.shape != truth.shape:
            raise ValueError('Paired ground truth alignment mismatch; do not silently resize')
        images = [('Input', target), ('Reference', reference), ('Paired GT', truth),
                  ('Current', baseline), ('Canon E2E', global_rgb), ('SA-LUT Log input' if args.native_log else 'SA-LUT', candidate)]
        panel = Image.new('RGB', (360*3, 260*2), '#222222')
        draw = ImageDraw.Draw(panel)
        for n, (label, rgb) in enumerate(images):
            im = Image.fromarray(rgb)
            im.thumbnail((356, 232))
            x, y = n%3*360, n//3*260
            panel.paste(im, (x+(360-im.width)//2, y+24+(232-im.height)//2))
            draw.text((x+8, y+6), label, fill='white')
            if label not in ('Reference', 'Paired GT'):
                Image.fromarray(rgb).save(out / f'{index}-{label.replace(" ", "_")}.png')
        panel.save(out / f'{index}-comparison.jpg', quality=94)
        row = {'index': index, 'target_shape': list(target.shape), 'diagnostics': diagnostic,
               'input': metrics(target, truth), 'current': metrics(baseline, truth),
               'canonical': metrics(global_rgb, truth), 'salut': metrics(candidate, truth)}
        rows.append(row)
        report = {'scope': 'First five PST50 development pairs only, RGB MAE/PSNR are not aesthetic judgment',
                  'upstream_commit': '3e62f9c9b59ea7b30d220ff6ba61047dab898c9a',
                  'checkpoint_sha256': CHECKPOINT_SHA, 'source_hashes': SOURCE_HASHES,
                  'algorithm_fingerprint': algorithm_fingerprint(),
                  'adapter_sha256': hashlib.sha256(Path(__file__).with_name('salut_model.py').read_bytes()).hexdigest(),
                  'interpolation_sha256': hashlib.sha256(Path(__file__).with_name('salut_interpolation.py').read_bytes()).hexdigest(),
                  'cases': rows}
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
