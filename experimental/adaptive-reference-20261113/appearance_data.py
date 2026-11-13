"""Frozen RGB8 cross-content validation controls for appearance supervision."""
import torch
from train_style_probe import fixed_validation_codes
from style_probe import render_style
from style_features import pooled_moments
from style_composition import compose_styles, fallback_unobserved_roles
from appearance_metrics import assess_appearance


def rgb8(x):
    return x.mul(255).round().clamp(0, 255).byte().float()/255


def control_pairs(seed=2026111307):
    reference = fixed_validation_codes(seed)
    target = fixed_validation_codes(seed+1)
    target[:4] = reference[:4]
    target[4:7] = 0
    return reference, target


def image_pairs(records):
    result = []
    for group in ('single', 'multiple', 'none'):
        ii = [i for i, r in enumerate(records) if r['group'] == group]
        if len(ii) != 10:
            raise ValueError('Expected ten heldout originals per group')
        result.extend(zip(ii[::2], ii[1::2]))
    return result


@torch.no_grad()
def build_validation(extractor, rgb, person, availability, records, seed=2026111307):
    source_codes, target_codes = control_pairs(seed)
    specs = [(target, reference, style) for style in range(10) for target, reference in image_pairs(records)]
    cache = {k: [] for k in ('reference_stats', 'reference_semantic', 'reference_person', 'target_stats',
                              'target_semantic', 'target_person', 'target_input', 'truth', 'human_mask')}
    metadata = []
    for start in range(0, len(specs), 8):
        chunk = specs[start:start+8]
        tt = torch.tensor([x[0] for x in chunk], device=rgb.device)
        rr = torch.tensor([x[1] for x in chunk], device=rgb.device)
        ss = torch.tensor([x[2] for x in chunk])
        ref_code, target_code = source_codes[ss].to(rgb.device), target_codes[ss].to(rgb.device)
        reference = rgb8(render_style(rgb[rr].float()/255, person[rr], ref_code))
        target = rgb8(render_style(rgb[tt].float()/255, person[tt], target_code))
        features = extractor.extract(torch.cat((reference, target)))
        n = len(chunk)
        for role, sl in (('reference', slice(0, n)), ('target', slice(n, None))):
            cache[role+'_stats'].append(features['stats'][sl])
            cache[role+'_semantic'].append(features['semantic'][sl])
            cache[role+'_person'].append(features['person'][sl])
        cache['target_input'].append(target)
        cache['truth'].append(rgb8(render_style(rgb[tt].float()/255, person[tt], ref_code)))
        cache['human_mask'].append(person[tt])
        for target_i, reference_i, style_i in chunk:
            group = records[target_i]['group']
            metadata.append({'target': records[target_i]['id'], 'reference': records[reference_i]['id'], 'group': group,
                             'style_index': style_i, 'same_style': style_i < 4,
                             'local_control': group != 'none' and style_i >= 5})
    return {k: torch.cat(v) for k, v in cache.items()}, metadata


def region_l1(output, truth, human_mask):
    weights = torch.cat((human_mask, 1-human_mask), dim=1)
    errors = (output-truth).abs().mean(dim=1, keepdim=True)
    mass = weights.sum(dim=(2, 3))
    regional = (errors*weights).sum(dim=(2, 3))/mass.clamp_min(1)
    visible = (mass > 1).float()
    return (regional*visible).sum()/visible.sum().clamp_min(1)


def relation_error(output, truth, human_mask):
    weights = torch.cat((human_mask, 1-human_mask), dim=1)
    actual, _ = pooled_moments(output, weights)
    expected, _ = pooled_moments(truth, weights)
    return ((actual[:, 0]-actual[:, 1])-(expected[:, 0]-expected[:, 1])).abs().mean(dim=1)*255


@torch.no_grad()
def validate_appearance(model, cache, metadata):
    was_training = model.training
    model.eval()
    rows = []
    for start in range(0, len(metadata), 8):
        sl = slice(start, start+8)
        before = model(cache['target_stats'][sl], cache['target_semantic'][sl])
        after = model(cache['reference_stats'][sl], cache['reference_semantic'][sl])
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
