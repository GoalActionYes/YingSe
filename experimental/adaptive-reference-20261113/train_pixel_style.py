"""Learn style-sensitive pixel features; keep the region detector frozen."""
import argparse
import json
import math
from pathlib import Path
import time
import uuid
import torch
from independent_coco import ROOT, digest
from train_style_probe import load_images
from train_style_appearance import sample_pairs
from style_probe import render_style, draw_style_codes, masked_parameter_mse
from style_composition import compose_styles, fallback_unobserved_roles
from appearance_data import rgb8, region_l1
from pixel_style_encoder import PixelFeatures, PixelStyleEncoder
from pixel_style_validation import build_pixel_cache, validate_pixel_style


def save_model(path, model, step, config_sha):
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'architecture': 'PixelStyleEncoder-v1', 'step': step, 'config_sha256': config_sha}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-start', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    steps, every = (8, 4) if args.smoke else (6000, 300)
    data = ROOT / 'outputs/adaptive-reference-20261113/supervised-style'
    manifest_path = data / 'manifest.json'
    warm = Path(args.warm_start).resolve()
    if not warm.is_relative_to((data / 'training').resolve()):
        raise ValueError('Expected own parameter pretraining run')
    warm_config = json.loads((warm / 'config.json').read_text(encoding='utf-8'))
    warm_report = json.loads((warm / 'results.json').read_text(encoding='utf-8'))
    if warm_config['stage'] != 'full' or warm_report['completed_steps'] != 1800 or warm_config['manifest_sha256'] != digest(manifest_path):
        raise ValueError('Invalid pretraining lineage')
    for name, expected in warm_config['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Pretraining source changed')
    torch.manual_seed(2026111303)
    run = data / 'pixel-training' / (time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    source_names = ('pixel_style_encoder.py', 'pixel_style_validation.py', 'train_pixel_style.py', 'pixel-style-protocol.md',
                    'style_features.py', 'style_probe.py', 'style_composition.py', 'appearance_data.py', 'appearance_metrics.py',
                    'train_style_probe.py', 'train_style_appearance.py', 'person_segmenter.py')
    config = {'scope': 'Synthetic appearance feature learning, not natural/video/product acceptance',
              'stage': 'smoke' if args.smoke else 'full', 'steps': steps, 'validation_every': every,
              'seed': 2026111303, 'validation_seed': 2026111307, 'batch_pairs': 4,
              'backbone_lr': .00001, 'head_lr': .0001, 'lr_final_fraction': .1,
              'weight_decay': .0001, 'appearance_weight': 10, 'parameter_weight': .2, 'same_style_weight': .01,
              'warm_config_sha256': digest(warm / 'config.json'), 'warm_checkpoint_sha256': digest(warm / 'best-semantic_features.pt'),
              'manifest_sha256': digest(manifest_path),
              'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in source_names}, 'test_content_loaded': False}
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    config_sha = digest(run / 'config.json')
    print(json.dumps({'run_directory': str(run), 'stage': config['stage']}), flush=True)
    device = torch.device('cuda:0')
    rgb, person, available, records = load_images(manifest_path, 'train', device)
    vrgb, vperson, vavailable, vrecords = load_images(manifest_path, 'validation', device)
    if {r['id'] for r in records} & {r['id'] for r in vrecords}:
        raise ValueError('Train/validation original identities overlap')
    extractor = PixelFeatures(str(device))
    cache, metadata = build_pixel_cache(extractor, vrgb, vperson, vavailable, vrecords)
    monitor_indices = []
    for group in ('single', 'multiple', 'none'):
        monitor_indices.extend([i for i, r in enumerate(records) if r['group'] == group][:10])
    mi = torch.tensor(monitor_indices, device=device)
    monitor_cache, monitor_metadata = build_pixel_cache(extractor, rgb[mi], person[mi], available[mi], [records[i] for i in monitor_indices])
    model = PixelStyleEncoder(extractor.model.backbone).to(device)
    payload = torch.load(warm / 'best-semantic_features.pt', weights_only=True, map_location='cpu')
    if payload['config_sha256'] != config['warm_config_sha256'] or payload['step'] != warm_report['best']['semantic_features']['step']:
        raise ValueError('Warm-start checkpoint identity mismatch')
    model.head.load_state_dict(payload['model'], strict=True)
    bn_before = {k: v.detach().clone() for k, v in model.backbone.state_dict().items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': .00001},
                                   {'params': model.head.parameters(), 'lr': .0001}], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: .1+.9*(1+math.cos(math.pi*step/steps))/2)
    generator = torch.Generator().manual_seed(2026111303)
    best_any, best_eligible, history = None, None, []
    started = time.perf_counter()
    for step in range(0, steps+1):
        if step:
            rr, tt = sample_pairs(records, generator, device)
            s, t = draw_style_codes(4, generator), draw_style_codes(4, generator)
            same = torch.rand(4, generator=generator) < .25
            t[same] = s[same]
            s, t = s.to(device), t.to(device)
            reference = rgb8(render_style(rgb[rr].float()/255, person[rr], s))
            target = rgb8(render_style(rgb[tt].float()/255, person[tt], t))
            truth = rgb8(render_style(rgb[tt].float()/255, person[tt], s))
            pixels = torch.cat((reference, target))
            features = extractor.extract(pixels)
            model.train()
            prediction = model(pixels, features['stats'], features['native_person'])
            predicted_reference = fallback_unobserved_roles(prediction[:4], features['person'][:4])
            predicted_target = fallback_unobserved_roles(prediction[4:], features['person'][4:])
            output = compose_styles(target, features['person'][4:], predicted_target, predicted_reference)
            appearance = ((output-truth).abs().mean()+region_l1(output, truth, person[tt]))/2
            parameter = masked_parameter_mse(prediction, torch.cat((s, t)), torch.cat((available[rr], available[tt])))
            common = available[rr]*available[tt]*same.to(device)[:, None]
            consistency = masked_parameter_mse(prediction[:4], prediction[4:], common)
            loss = 10*appearance+.2*parameter+.01*consistency
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite pixel-style loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if step%100 == 0 or step == 1:
                print(json.dumps({'step': step, 'seconds': time.perf_counter()-started, 'training_mae': float(appearance.detach()*255),
                                  'parameter_mse': float(parameter.detach())}), flush=True)
        if step%every == 0 or step == steps:
            gate, rows = validate_pixel_style(model, cache, metadata)
            monitor, _ = validate_pixel_style(model, monitor_cache, monitor_metadata)
            current = {'step': step, 'gate': gate}
            if best_any is None or gate['mean_output_mae'] < best_any['gate']['mean_output_mae']:
                best_any = current
                save_model(run / 'best.pt', model, step, config_sha)
                (run / 'best-validation.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            if gate['passed'] and (best_eligible is None or gate['mean_output_mae'] < best_eligible['gate']['mean_output_mae']):
                best_eligible = current
                save_model(run / 'eligible.pt', model, step, config_sha)
                (run / 'eligible-validation.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            history.append({'step': step, 'seconds': time.perf_counter()-started, 'validation': gate, 'training_monitor': monitor})
            report = {'config': config, 'feature_provenance': extractor.provenance, 'completed_steps': step,
                      'history': history, 'best_any': best_any, 'best_eligible': best_eligible,
                      'test_content_loaded': False,
                      'frozen_detector_trainable_parameters': sum(p.numel() for p in extractor.model.parameters() if p.requires_grad)}
            (run / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'validation_step': step, 'gate': gate, 'training_monitor_mae': monitor['mean_output_mae']}), flush=True)
    after = model.backbone.state_dict()
    if any(not torch.equal(value, after[name]) for name, value in bn_before.items()):
        raise ValueError('Style-backbone BN running statistics changed')
    if any(p.requires_grad for p in extractor.model.parameters()):
        raise ValueError('Frozen region detector became trainable')
    print(json.dumps({'finished': True, 'run_directory': str(run), 'best_eligible': best_eligible, 'best_any': best_any}), flush=True)


if __name__ == '__main__':
    main()
