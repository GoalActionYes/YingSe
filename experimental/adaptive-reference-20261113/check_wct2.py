"""Five already inspected development pairs, resized consistently, no selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from wct2_model import ROOT, WCT2Model, HASHES
from check_salut import metrics


def working_rgb(path, size):
    return np.array(Image.open(path).convert('RGB').resize(size, Image.Resampling.BILINEAR))


def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/pst50'
    out = data / 'development-wct2'
    out.mkdir(exist_ok=True)
    model = WCT2Model(max_edge=640)
    rows = []
    for index in range(1, 6):
        target = np.array(Image.open(data / f'content_709/in{index}.png').convert('RGB'))
        reference = np.array(Image.open(data / f'paired_style/tar{index}.png').convert('RGB'))
        output, diagnostics = model.predict(target, reference)
        size = output.shape[1], output.shape[0]
        # Paired target appears only after prediction; other methods use the same resize.
        truth = working_rgb(data / f'paired_gt/gt{index}.png', size)
        current = working_rgb(data / f'development-salut/{index}-Current.png', size)
        canonical = working_rgb(data / f'development-salut/{index}-Canon_E2E.png', size)
        target_working = np.array(Image.fromarray(target).resize(size, Image.Resampling.BILINEAR))
        reference_working = np.array(Image.fromarray(reference).resize(size, Image.Resampling.BILINEAR))
        row = {'index': index, 'current': metrics(current, truth), 'canonical': metrics(canonical, truth),
               'wct2': metrics(output, truth), 'diagnostics': diagnostics}
        rows.append(row)
        Image.fromarray(output).save(out / f'{index}-wct2.png')
        panel = Image.new('RGB', (360*3, 260*2), '#222222')
        draw = ImageDraw.Draw(panel)
        for n, (label, rgb) in enumerate([('Input RGB', target_working), ('Reference', reference_working), ('Paired GT', truth),
                                         ('Current', current), ('Canon E2E', canonical), ('WCT2', output)]):
            im = Image.fromarray(rgb)
            im.thumbnail((356, 232))
            x, y = n%3*360, n//3*260
            panel.paste(im, (x+(360-im.width)//2, y+24+(232-im.height)//2))
            draw.text((x+8, y+6), label, fill='white')
        panel.save(out / f'{index}-comparison.jpg', quality=94)
        print(json.dumps(row), flush=True)
        report = {'scope': 'First five development pairs; every method/GT scored at same 640x360 working resolution',
                  'upstream_commit': '05d17c02c6eb78a3e0afdbc1cf1d7549d4ad1543', 'source_and_weight_sha256': HASHES,
                  'adapter_sha256': hashlib.sha256(Path(__file__).with_name('wct2_model.py').read_bytes()).hexdigest(),
                  'protocol_sha256': hashlib.sha256(Path(__file__).with_name('wct2-protocol.md').read_bytes()).hexdigest(), 'cases': rows}
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Raw same-image behavior before any product identity guard.
    original = np.array(Image.open(data / 'content_709/in1.png').convert('RGB'))
    same, timing = model.predict(original, original)
    resized = np.array(Image.fromarray(original).resize((same.shape[1], same.shape[0]), Image.Resampling.BILINEAR))
    report['raw_same_image'] = {'scores': metrics(same, resized), 'diagnostics': timing}
    Image.fromarray(same).save(out / 'raw-same-image.png')
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'raw_same_image': report['raw_same_image']}), flush=True)


if __name__ == '__main__':
    main()
