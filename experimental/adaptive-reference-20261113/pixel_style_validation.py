"""Same validation controls and gates, with RGB inputs for the learned backbone."""
import torch
from appearance_data import build_validation, control_pairs, rgb8, relation_error
from style_probe import render_style
from style_composition import compose_styles, fallback_unobserved_roles
from appearance_metrics import assess_appearance


@torch.no_grad()
def build_pixel_cache(extractor, rgb, person, availability, records):
    cache, metadata = build_validation(extractor, rgb, person, availability, records)
    controls, _ = control_pairs()
    by_id = {r['id']: i for i, r in enumerate(records)}
    refs, ref_native, target_native = [], [], []
    for start in range(0, len(metadata), 8):
        info = metadata[start:start+8]
        ii = torch.tensor([by_id[r['reference']] for r in info], device=rgb.device)
        styles = torch.tensor([r['style_index'] for r in info])
        reference = rgb8(render_style(rgb[ii].float()/255, person[ii], controls[styles].to(rgb.device)))
        features = extractor.extract(torch.cat((reference, cache['target_input'][start:start+len(info)])))
        refs.append(reference)
        ref_native.append(features['native_person'][:len(info)])
        target_native.append(features['native_person'][len(info):])
    cache.update(reference_rgb=torch.cat(refs), reference_native=torch.cat(ref_native), target_native=torch.cat(target_native))
    return cache, metadata


@torch.no_grad()
def validate_pixel_style(model, cache, metadata):
    was_training = model.training
    model.eval()
    rows = []
    for start in range(0, len(metadata), 8):
        sl = slice(start, start+8)
        before = model(cache['target_input'][sl], cache['target_stats'][sl], cache['target_native'][sl])
        after = model(cache['reference_rgb'][sl], cache['reference_stats'][sl], cache['reference_native'][sl])
        before = fallback_unobserved_roles(before, cache['target_person'][sl])
        after = fallback_unobserved_roles(after, cache['reference_person'][sl])
        inputs, truth, mask = cache['target_input'][sl], cache['truth'][sl], cache['human_mask'][sl]
        output = rgb8(compose_styles(inputs, cache['target_person'][sl], before, after))
        input_errors = (inputs-truth).abs().mean(dim=(1, 2, 3))*255
        errors = (output-truth).abs().mean(dim=(1, 2, 3))*255
        rb, ra = relation_error(inputs, truth, mask), relation_error(output, truth, mask)
        for i, info in enumerate(metadata[sl]):
            rows.append(dict(info, input_mae=float(input_errors[i]), output_mae=float(errors[i]),
                             input_relation_error=float(rb[i]) if info['local_control'] else None,
                             output_relation_error=float(ra[i]) if info['local_control'] else None))
    model.train(was_training)
    return assess_appearance(rows), rows
