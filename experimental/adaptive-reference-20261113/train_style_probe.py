"""Reference-only supervised identifiability probe; NEVER loads test originals.

Two regressors see the same styles/content stream. Frozen segmentation features
accept RGB only; human masks are used only by teacher synthesis and losses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image, ImageFilter
import torch
from style_probe import StyleRegressor, render_style, draw_style_codes, masked_parameter_mse, PARAM_SCALE
from style_features import StyleFeatures
from independent_coco import ROOT, digest


def load_images(manifest_path, split, device):
    if split not in ('train', 'validation'):
        raise ValueError('Training process must not load final test content')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = [r for r in manifest['images'] if r['split'] == split]
    if len(records) != {'train': 240, 'validation': 30}[split]:
        raise ValueError('Unexpected split size')
    images, masks = [], []
    for record in records:
        image_path, mask_path = manifest_path.parent / record['image_file'], manifest_path.parent / record['mask_file']
        if digest(image_path) != record['image_sha256'] or digest(mask_path) != record['mask_sha256']:
            raise ValueError('Prepared data changed')
        images.append(np.array(Image.open(image_path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)))
        masks.append(np.array(Image.open(mask_path).convert('L').resize((224, 224), Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(2))))
    rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().to(device)
    person = torch.from_numpy(np.stack(masks))[:, None].to(device=device, dtype=torch.float32)/255
    availability = torch.stack(((person.mean(dim=(1, 2, 3)) > .005).float(), ((1-person).mean(dim=(1, 2, 3)) > .005).float()), dim=1)
    return rgb, person, availability, records


def fixed_validation_codes(seed=2026111301):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.rand(10, 2, 6, generator=generator)*2-1
    codes[:2] = 0
    codes[2:5, 1] = codes[2:5, 0]
    return codes


@torch.no_grad()
def validation_features(extractor, rgb, person, availability):
    codes = fixed_validation_codes().to(rgb.device).repeat_interleave(rgb.shape[0], dim=0)
    indices = torch.arange(rgb.shape[0], device=rgb.device).repeat(10)
    stats, semantic = [], []
    for start in range(0, len(indices), 8):
        ii = indices[start:start+8]
        styled = render_style(rgb[ii].float()/255, person[ii], codes[start:start+8])
        features = extractor.extract(styled)
        stats.append(features['stats'])
        semantic.append(features['semantic'])
    return {'stats': torch.cat(stats), 'semantic': torch.cat(semantic), 'truth': codes,
            'availability': availability[indices], 'base_count': rgb.shape[0]}


@torch.no_grad()
def validate(model, cached):
    model.eval()
    prediction = model(cached['stats'], cached['semantic'])
    mse = masked_parameter_mse(prediction, cached['truth'], cached['availability'])
    zero = masked_parameter_mse(torch.zeros_like(prediction), cached['truth'], cached['availability'])
    weighted = (prediction-cached['truth']).square()*cached['availability'][:, :, None]
    per_parameter = weighted.sum(dim=(0, 1))/cached['availability'].sum().clamp_min(1)
    # Spread among different contents with exactly the same known style.
    spread = []
    for start in range(0, prediction.shape[0], cached['base_count']):
        block = prediction[start:start+cached['base_count']]
        weights = cached['availability'][start:start+cached['base_count']]
        mean = (block*weights[:, :, None]).sum(dim=0)/weights.sum(dim=0).clamp_min(1)[:, None]
        spread.append(((block-mean).square()*weights[:, :, None]).sum()/weights.sum().clamp_min(1)/6)
    model.train()
    return {'normalized_parameter_mse': float(mse), 'zero_parameter_mse': float(zero),
            'relative_mse_to_zero': float(mse/zero.clamp_min(1e-9)),
            'per_parameter_mse': per_parameter.cpu().tolist(), 'same_style_cross_content_variance': float(torch.stack(spread).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    steps, every = (8, 4) if args.smoke else (1800, 150)
    torch.manual_seed(20261113)
    np.random.seed(20261113)
    data = ROOT / 'outputs/adaptive-reference-20261113/supervised-style'
    manifest_path = data / 'manifest.json'
    run = data / 'training' / (time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    source_names = ('style_dataset.py', 'style_probe.py', 'style_features.py', 'person_segmenter.py',
                    'train_style_probe.py', 'supervised-style-probe.md')
    config = {'scope': 'Supervised style identifiability only; no product transfer or natural-aesthetic claim',
              'stage': 'smoke' if args.smoke else 'full', 'steps': steps, 'batch': 8, 'style_pairs_per_batch': 4,
              'learning_rate': .001, 'weight_decay': .0001, 'consistency_weight': .1,
              'seed': 20261113, 'validation_seed': 2026111301, 'validation_every': every,
              'parameter_scale': list(PARAM_SCALE), 'manifest_sha256': digest(manifest_path),
              'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in source_names},
              'test_content_loaded': False}
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'run_directory': str(run), 'stage': config['stage']}), flush=True)
    device = torch.device('cuda:0')
    rgb, person, availability, train_records = load_images(manifest_path, 'train', device)
    vrgb, vperson, vavailable, validation_records = load_images(manifest_path, 'validation', device)
    if {r['id'] for r in train_records} & {r['id'] for r in validation_records}:
        raise ValueError('Training and validation identities overlap')
    extractor = StyleFeatures(device=str(device))
    cached = validation_features(extractor, vrgb, vperson, vavailable)
    dimensions = cached['stats'].shape[1], cached['semantic'].shape[1]
    models, optimizers, best = {}, {}, {}
    for name, semantic in (('color_statistics', False), ('semantic_features', True)):
        torch.manual_seed(20261113)
        models[name] = StyleRegressor(*dimensions, use_semantic=semantic).to(device)
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=.001, weight_decay=.0001)
        best[name] = {'loss': float('inf')}
    training_generator = torch.Generator().manual_seed(20261113)
    rows = []
    started = time.perf_counter()
    for step in range(1, steps+1):
        first = torch.randint(len(rgb), (4,), generator=training_generator)
        second = (first+torch.randint(1, len(rgb), (4,), generator=training_generator))%len(rgb)
        ii = torch.stack((first, second), dim=1).flatten().to(device)
        codes = draw_style_codes(4, training_generator).repeat_interleave(2, dim=0).to(device)
        styled = render_style(rgb[ii].float()/255, person[ii], codes)
        features = extractor.extract(styled)
        weights = availability[ii]
        losses = {}
        for name, model in models.items():
            model.train()
            prediction = model(features['stats'], features['semantic'])
            supervised = masked_parameter_mse(prediction, codes, weights)
            common = weights[::2]*weights[1::2]
            consistent = masked_parameter_mse(prediction[::2], prediction[1::2], common)
            loss = supervised+.1*consistent
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizers[name].step()
            losses[name] = float(supervised.detach())
        if step%50 == 0 or step == 1:
            print(json.dumps({'step': step, 'seconds': time.perf_counter()-started, 'training_mse': losses}), flush=True)
        if step%every == 0 or step == steps:
            results = {name: validate(model, cached) for name, model in models.items()}
            row = {'step': step, 'seconds': time.perf_counter()-started, 'training_mse': losses, 'validation': results}
            rows.append(row)
            for name, result in results.items():
                if result['normalized_parameter_mse'] < best[name]['loss']:
                    best[name] = {'loss': result['normalized_parameter_mse'], 'step': step, 'validation': result}
                    payload = {'model': {k: v.detach().cpu() for k, v in models[name].state_dict().items()},
                               'stats_dim': dimensions[0], 'semantic_dim': dimensions[1], 'use_semantic': name == 'semantic_features',
                               'step': step, 'config_sha256': digest(run / 'config.json')}
                    torch.save(payload, run / f'best-{name}.pt')
            report = {'config': config, 'feature_provenance': extractor.provenance, 'completed_steps': step,
                      'validation_history': rows, 'best': best, 'test_content_loaded': False,
                      'frozen_backbone_trainable_parameters': sum(p.numel() for p in extractor.model.parameters() if p.requires_grad)}
            (run / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'validation_step': step, 'results': results}), flush=True)
    if extractor.model.training or any(p.requires_grad for p in extractor.model.parameters()):
        raise ValueError('Frozen feature extractor state changed')
    print(json.dumps({'finished': True, 'run_directory': str(run), 'best': best}), flush=True)


if __name__ == '__main__':
    main()
