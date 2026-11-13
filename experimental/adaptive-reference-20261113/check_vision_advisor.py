"""Run fixed first-five public development pairs; never use held-out images."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from person_segmenter import PersonSegmenter
from vision_advisor import ROOT, VisionAdvisor, render_plan, file_sha, MODEL_REVISION


def load_rgb(path, size=None):
    with Image.open(path) as im:
        im = im.convert('RGB')
        if size:
            im = im.resize(size,Image.Resampling.BILINEAR)
        else:
            im.thumbnail((640,640),Image.Resampling.BILINEAR)
        return np.asarray(im).copy()


def metric(rgb, truth):
    return float(np.abs(rgb.astype(np.float32)-truth.astype(np.float32)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('--categories',action='store_true')
    args = parser.parse_args()
    data = ROOT/'outputs/adaptive-reference-20261113/pst50'
    out = data/(args.out or ('development-vision-categories-v1' if args.categories else 'development-vision-advisor-v1'))
    out.mkdir(exist_ok=False)
    sources = ['vision_advisor.py','check_vision_advisor.py','vision-advisor-protocol.md','person_segmenter.py']
    if args.categories: sources += ['vision_categories.py','vision-categorical-protocol.md']
    frozen = out/'frozen-source'; frozen.mkdir()
    hashes = {}
    for name in sources:
        source = Path(__file__).with_name(name)
        (frozen/name).write_bytes(source.read_bytes())
        hashes[name] = file_sha(source)
    report = {'scope':'PST50 1-5 development only, no held-out consumption; RGB MAE is not aesthetics',
        'revision':MODEL_REVISION,'source_sha256':hashes,'cases':[]}
    report_path = out/'results.json'
    def save(): report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    print('Loading local pinned model',flush=True)
    advisor = VisionAdvisor()
    print(f'Model loaded in {advisor.load_seconds:.3f}s',flush=True)
    segmenter = PersonSegmenter(device='cuda')
    report['load_seconds'] = advisor.load_seconds
    report['segmentation'] = segmenter.provenance
    save()
    for index in range(1,8 if args.categories else 7):
        same = index == 6
        controlled = index == 7
        n = 1 if same or controlled else index
        target_path = data/f'content_709/in{n}.png'
        reference_path = target_path if same else data/f'paired_style/tar{n}.png'
        target, reference = load_rgb(target_path),load_rgb(reference_path)
        if controlled:
            reference = np.rint(target.astype(np.float32)*np.array([.60,.75,.95],dtype=np.float32)).astype(np.uint8)
            Image.fromarray(reference).save(out/'controlled-reference.png')
        print(f'Advising {index} same={same}',flush=True)
        if args.categories:
            from vision_categories import CATEGORY_PROMPT, parse_categories
            plan, diagnostics = advisor.advise(reference,target,prompt=CATEGORY_PROMPT,parser=parse_categories)
        else:
            plan, diagnostics = advisor.advise(reference,target)
        row = {'index':index,'same_image':same,'diagnostics':diagnostics,
            'target_sha256':file_sha(target_path),'reference_sha256':file_sha(reference_path),
            'working_shape':list(target.shape)}
        row['controlled_same_content'] = controlled
        if controlled:
            row['reference_sha256'] = file_sha(out/'controlled-reference.png')
        report['cases'].append(row); save()
        if plan is None:
            print(json.dumps(row),flush=True)
            continue
        mask, mask_diagnostics = segmenter.predict(target)
        row['segmentation_diagnostics'] = mask_diagnostics
        row['mask_mean'] = float(mask.mean())
        applied = plan.copy()
        if mask.mean() < .02:
            applied[0] = applied[1]
            row['person_fallback_to_background'] = True
        row['applied_plan'] = applied.tolist()
        start = time.perf_counter()
        output = render_plan(target,mask,applied)
        row['render_seconds'] = time.perf_counter()-start
        Image.fromarray(output).save(out/f'{index}-advisor.png')
        Image.fromarray(np.rint(mask*255).astype(np.uint8)).save(out/f'{index}-mask.png')
        size = (target.shape[1],target.shape[0])
        # Paired desired image is read only after advice, segmentation and rendering.
        truth = reference if controlled else target if same else load_rgb(data/f'paired_gt/gt{n}.png',size)
        current = target if same or controlled else load_rgb(data/f'development-salut/{n}-Current.png',size)
        row['input_mae'] = metric(target,truth)
        row['current_mae'] = None if controlled else metric(current,truth)
        row['advisor_mae'] = metric(output,truth)
        panel = Image.new('RGB',(360*3,260*2),'#222222')
        draw = ImageDraw.Draw(panel)
        for k,(label,rgb) in enumerate([('Input',target),('Reference',reference),('Expert target',truth),
                                         ('Current',current),('Vision advisor',output),('Automatic mask',np.repeat(np.rint(mask[...,None]*255).astype(np.uint8),3,axis=2))]):
            im = Image.fromarray(rgb); im.thumbnail((356,232))
            x,y = k%3*360,k//3*260
            panel.paste(im,(x+(360-im.width)//2,y+24+(232-im.height)//2))
            draw.text((x+8,y+6),label,fill='white')
        panel.save(out/f'{index}-comparison.jpg',quality=94)
        save(); print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
