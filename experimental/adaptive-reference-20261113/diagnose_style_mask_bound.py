"""Validation-only ORACLE-style diagnostic; never a deployable candidate.

Known S/T codes are supplied to isolate mask/renderer limits from style inference.
Scores must not be reported as learned-model quality or final-test performance.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from independent_coco import ROOT, digest
from train_style_probe import load_images
from style_probe import render_style
from style_features import StyleFeatures
from style_composition import compose_styles, fallback_unobserved_roles
from appearance_data import build_validation, control_pairs, rgb8, relation_error
from appearance_metrics import assess_appearance


@torch.no_grad()
def probability_520(extractor, rgb):
    resized = F.interpolate(rgb, size=(520, 520), mode='bilinear', align_corners=False, antialias=True)
    mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    logits = extractor.model((resized-mean)/std)['out']
    probability = logits.softmax(dim=1)[:, 15:16]
    return F.interpolate(probability, size=(224, 224), mode='bilinear', align_corners=False)


@torch.no_grad()
def main():
    data = ROOT / 'outputs/adaptive-reference-20261113/supervised-style'
    rgb, person, available, records = load_images(data / 'manifest.json', 'validation', 'cuda:0')
    extractor = StyleFeatures('cuda:0')
    cache, metadata = build_validation(extractor, rgb, person, available, records)
    ref_controls, target_controls = control_pairs()
    indices = {r['id']: i for i, r in enumerate(records)}
    rows = {k: [] for k in ('oracle_human_mask', 'oracle_auto_224', 'oracle_auto_520')}
    for start in range(0, len(metadata), 4):
        sl = slice(start, start+4)
        info = metadata[sl]
        codes = torch.tensor([r['style_index'] for r in info])
        s, t = ref_controls[codes].cuda(), target_controls[codes].cuda()
        inputs, truth, human = cache['target_input'][sl], cache['truth'][sl], cache['human_mask'][sl]
        rr = torch.tensor([indices[r['reference']] for r in info], device='cuda')
        reference = rgb8(render_style(rgb[rr].float()/255, person[rr], s))
        reference_520 = probability_520(extractor, reference)
        target_520 = probability_520(extractor, inputs)
        outputs = {
            'oracle_human_mask': compose_styles(inputs, human, t, s),
            'oracle_auto_224': compose_styles(inputs, cache['target_person'][sl],
                                              fallback_unobserved_roles(t, cache['target_person'][sl]),
                                              fallback_unobserved_roles(s, cache['reference_person'][sl])),
            'oracle_auto_520': compose_styles(inputs, target_520, fallback_unobserved_roles(t, target_520),
                                              fallback_unobserved_roles(s, reference_520)),
        }
        input_mae = (inputs-truth).abs().mean(dim=(1, 2, 3))*255
        rb = relation_error(inputs, truth, human)
        for name, output in outputs.items():
            output = rgb8(output)
            mae = (output-truth).abs().mean(dim=(1, 2, 3))*255
            ra = relation_error(output, truth, human)
            for i, item in enumerate(info):
                rows[name].append(dict(item, input_mae=float(input_mae[i]), output_mae=float(mae[i]),
                                       input_relation_error=float(rb[i]) if item['local_control'] else None,
                                       output_relation_error=float(ra[i]) if item['local_control'] else None))
    report = {'scope': 'Validation-only known-style oracle; isolates rendering/masks, NOT inferred-style or product quality',
              'model_style_codes_are_oracle': True, 'test_content_loaded': False,
              'manifest_sha256': digest(data / 'manifest.json'), 'source_sha256': digest(Path(__file__)),
              'summaries': {k: assess_appearance(v) for k, v in rows.items()}, 'cases': rows}
    (data / 'validation-oracle-mask-bound.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summaries'], indent=2), flush=True)


if __name__ == '__main__':
    main()
