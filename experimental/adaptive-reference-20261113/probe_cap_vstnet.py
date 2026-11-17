"""Screen CAP-VSTNet with the locally pinned ADE20K SegFormer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from probe_reference_chroma_spread import (
    BACKGROUND, BASE, GLOBAL, PARENT, decision, digest, projection, read, region_delta, target_regions,
)
from spatial_render import finish_spatial

ROOT = BASE.parents[1]
ASSETS = ROOT / '.tools/CAP-VSTNet-audit'
SEG = ROOT / '.tools/scene-segformer'
WEIGHT_SHA = 'e1b87d4d3cae8d59b2b91cb2471bc1b6777f3446b32b9a87bfd0021393038e1d'
PARENT_SHA = 'd7dac2996d03cdf8d5a08c3e0200272772bfd8c6f7fe0d81931c84db01cd33dc'
BLOBS = {
    'models/RevResNet.py': '467f987f802b7b7998eca4ac2cd24ff1f17be466',
    'models/cWCT.py': 'c9ebf94cfecebd8b86b8105b5e316ab9f3ab961b',
    'models/segmentation/SegReMapping.py': 'a45b6d22ff80b5a1a3ad46370b313f34be3a3150',
    'models/segmentation/ade20k_semantic_rel.npy': 'a650f67733a22e88c7bd5aa0df8c7e5cbf2aa972',
}


def load_candidate(device, replay=False):
    for name, expected in BLOBS.items():
        data = (ASSETS / name).read_bytes()
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == expected
    ckpt = ASSETS / 'photo_image.pt'
    assert digest(ckpt) == WEIGHT_SHA
    manifest = json.loads((SEG / 'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert digest(SEG / row['path']) == row['sha256']
    sys.path.insert(0, str(ASSETS))
    from models.RevResNet import RevResNet
    from models.cWCT import cWCT
    from models.segmentation.SegReMapping import SegReMapping

    model = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256],
                      in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict'], strict=True)
    model.to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(SEG, local_files_only=True)
    segmenter, info = SegformerForSemanticSegmentation.from_pretrained(
        SEG, local_files_only=True, use_safetensors=True, output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    segmenter.to(device).eval()
    transform = cWCT()

    def encode(rgb):
        features = model(torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float() / 255, forward=True)
        logits = segmenter(**processor(images=rgb, return_tensors='pt').to(device)).logits
        labels = F.interpolate(logits, size=rgb.shape[:2], mode='bilinear', align_corners=False)[0].argmax(0)
        return features, labels.to(torch.uint8).cpu().numpy()

    def render(target_features, target_labels, ref_features, ref_labels):
        remap = SegReMapping(ASSETS / 'models/segmentation/ade20k_semantic_rel.npy', min_ratio=.02)
        target_seg = remap.self_remapping(target_labels)
        ref_seg = remap.self_remapping(ref_labels)
        target_seg = remap.cross_remapping(target_seg, ref_seg)
        mapped = transform.transfer(target_features, ref_features, target_seg[None], ref_seg[None])
        result = model(mapped, forward=False)
        assert torch.isfinite(result).all()
        return (result[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()

    if replay:
        return encode, render, manifest['revision']

    def apply(target_features, target_labels, reference):
        return render(target_features, target_labels, *encode(reference))

    return encode, apply, manifest['revision']


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = PARENT / 'results.json'
    assert digest(parent) == PARENT_SHA
    baseline = json.loads(parent.read_text(encoding='utf-8'))
    assert baseline['strength'] == .8 and len(baseline['cases']) == 6
    encode, apply, revision = load_candidate('cuda')
    out.mkdir(exist_ok=False)
    cases = []
    start = time.monotonic()
    for case in baseline['cases']:
        pair = case['pair']
        folder = out / pair
        folder.mkdir()
        target = read(case['input_file'])
        target_encoding = encode(target)
        mask_name = pair.split('-')[0] + '-human-mask.png'
        path = BASE / 'independent-coco' / mask_name
        assert digest(path) == case['parents'][mask_name]
        with Image.open(path) as image:
            person = np.asarray(image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        masks = target_regions(target, person)
        variants = {v['name']: v for v in case['variants']}
        results = {}
        for name in ['identity', *BACKGROUND, *GLOBAL]:
            row = variants[name]
            native = apply(*target_encoding, read(row['reference']))
            outputs = dict(baseline=read(row['methods']['adaptive']['file']),
                           candidate=finish_spatial(target, native, .8))
            path = folder / f'{name}-candidate.png'
            Image.fromarray(outputs['candidate']).save(path)
            if name == 'identity':
                origin = outputs
            changes = {label: image.astype(np.int16) - origin[label] for label, image in outputs.items()}
            results[name] = dict(
                output_sha256=digest(path),
                regions={k: {r: region_delta(v, origin[k], m) for r, m in masks.items()}
                         for k, v in outputs.items()},
                projection={r: projection(changes['candidate'], changes['baseline'], m) for r, m in masks.items()},
            )
            assert time.monotonic() - start < 180
        cases.append(dict(pair=pair, variants=results))
        print(pair, flush=True)
    report = dict(cases=cases, decision=decision(cases), seconds=time.monotonic()-start,
                  parent_sha256=PARENT_SHA, cap_weight_sha256=WEIGHT_SHA, segmentation_revision=revision,
                  source_blobs=BLOBS, script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('cap-vstnet-screen-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['decision']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
