"""Appearance-supervised reference/target composition; final test never loaded."""
import argparse
import json
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from independent_coco import ROOT, digest
from train_style_probe import load_images
from style_probe import StyleRegressor, render_style, draw_style_codes, masked_parameter_mse
from style_features import StyleFeatures
from style_composition import compose_styles, fallback_unobserved_roles
from appearance_data import rgb8, build_validation, region_l1, validate_appearance


def sample_pairs(records, generator, device):
    pools = [[i for i, r in enumerate(records) if r['group'] == 'none'],
             [i for i, r in enumerate(records) if r['group'] != 'none']]
    references, targets = [], []
    for _ in range(4):
        pool = pools[int(torch.rand((), generator=generator) < 2/3)]
        first = int(torch.randint(len(pool), (), generator=generator))
        second = (first+int(torch.randint(1, len(pool), (), generator=generator)))%len(pool)
        references.append(pool[first])
        targets.append(pool[second])
    return torch.tensor(references, device=device), torch.tensor(targets, device=device)


def save_checkpoint(path, model, dimensions, name, step, config_sha):
    payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'stats_dim': dimensions[0], 'semantic_dim': dimensions[1],
               'use_semantic': name == 'semantic_features', 'step': step, 'config_sha256': config_sha}
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--warm-start')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--minimum-learning-rate', type=float, default=.00005)
    parser.add_argument('--parameter-weight', type=float, default=.02)
    args = parser.parse_args()
    if not 0 < args.minimum_learning_rate <= args.learning_rate or args.parameter_weight < 0:
        raise ValueError('Invalid optimizer configuration')
    steps, every = (8, 4) if args.smoke else (3600, 200)
    torch.manual_seed(2026111302)
    np.random.seed(2026111302)
    data = ROOT / 'outputs/adaptive-reference-20261113/supervised-style'
    manifest_path = data / 'manifest.json'
    run = data / 'appearance-training' / (time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    names = ('style_probe.py', 'style_features.py', 'style_composition.py', 'appearance_data.py', 'appearance_metrics.py',
             'train_style_appearance.py', 'train_style_probe.py', 'person_segmenter.py', 'appearance-training-protocol.md')
    warm = None
    if args.warm_start:
        warm = Path(args.warm_start).resolve()
        if not warm.is_relative_to((data / 'training').resolve()):
            raise ValueError('Expected own completed parameter-training directory')
        warm_config = json.loads((warm / 'config.json').read_text(encoding='utf-8'))
        warm_report = json.loads((warm / 'results.json').read_text(encoding='utf-8'))
        if warm_config['stage'] != 'full' or warm_report['completed_steps'] != 1800 or warm_config['manifest_sha256'] != digest(manifest_path):
            raise ValueError('Invalid parameter pretraining provenance')
        for name, expected in warm_config['source_sha256'].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise ValueError('Pretraining source changed: '+name)
        names += ('appearance-optimization-protocol.md',)
    config = {'scope': 'Synthetic cross-content appearance composition, not product acceptance',
              'stage': 'smoke' if args.smoke else 'full', 'steps': steps, 'batch_pairs': 4,
              'seed': 2026111302, 'validation_seed': 2026111307, 'validation_every': every,
              'learning_rate': args.learning_rate, 'minimum_learning_rate': args.minimum_learning_rate, 'weight_decay': .0001,
              'relative_identity_probability': .25, 'appearance_weight': 10, 'parameter_weight': args.parameter_weight,
              'same_style_weight': .01, 'manifest_sha256': digest(manifest_path),
              'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in names}, 'test_content_loaded': False}
    if warm:
        config['warm_start'] = {'config_sha256': digest(warm / 'config.json'),
                                'weights_sha256': {name: digest(warm / f'best-{name}.pt') for name in ('color_statistics', 'semantic_features')}}
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    config_sha = digest(run / 'config.json')
    print(json.dumps({'run_directory': str(run), 'stage': config['stage']}), flush=True)
    device = torch.device('cuda:0')
    rgb, person, availability, records = load_images(manifest_path, 'train', device)
    vrgb, vperson, vavailable, vrecords = load_images(manifest_path, 'validation', device)
    if {r['id'] for r in records} & {r['id'] for r in vrecords}:
        raise ValueError('Original identity leakage')
    extractor = StyleFeatures(str(device))
    cache, metadata = build_validation(extractor, vrgb, vperson, vavailable, vrecords)
    dimensions = cache['target_stats'].shape[1], cache['target_semantic'].shape[1]
    models, optimizers, schedulers = {}, {}, {}
    best_any, best_eligible = {}, {}
    for name, semantic in (('color_statistics', False), ('semantic_features', True)):
        torch.manual_seed(2026111302)
        models[name] = StyleRegressor(*dimensions, use_semantic=semantic).to(device)
        if warm:
            payload = torch.load(warm / f'best-{name}.pt', weights_only=True, map_location='cpu')
            if (payload['config_sha256'] != digest(warm / 'config.json') or payload['use_semantic'] != semantic
                    or payload['step'] != warm_report['best'][name]['step']):
                raise ValueError('Warm-start checkpoint/config mismatch')
            models[name].load_state_dict(payload['model'], strict=True)
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=args.learning_rate, weight_decay=.0001)
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[name], T_max=steps, eta_min=args.minimum_learning_rate)
    generator = torch.Generator().manual_seed(2026111302)
    history = []
    started = time.perf_counter()
    if warm:
        initial_gates = {}
        for name, model in models.items():
            initial_gates[name], initial_rows = validate_appearance(model, cache, metadata)
            best_any[name] = {'step': 0, 'gate': initial_gates[name]}
            save_checkpoint(run / f'best-{name}.pt', model, dimensions, name, 0, config_sha)
            (run / f'best-{name}-validation.json').write_text(json.dumps(initial_rows, indent=2), encoding='utf-8')
            if initial_gates[name]['passed']:
                best_eligible[name] = best_any[name]
                save_checkpoint(run / f'eligible-{name}.pt', model, dimensions, name, 0, config_sha)
                (run / f'eligible-{name}-validation.json').write_text(json.dumps(initial_rows, indent=2), encoding='utf-8')
        history.append({'step': 0, 'seconds': time.perf_counter()-started, 'gates': initial_gates})
        print(json.dumps({'validation_step': 0, 'gates': initial_gates}), flush=True)
    for step in range(1, steps+1):
        rr, tt = sample_pairs(records, generator, device)
        ref_code = draw_style_codes(4, generator)
        target_code = draw_style_codes(4, generator)
        same = torch.rand(4, generator=generator) < .25
        target_code[same] = ref_code[same]
        ref_code, target_code = ref_code.to(device), target_code.to(device)
        reference = rgb8(render_style(rgb[rr].float()/255, person[rr], ref_code))
        target = rgb8(render_style(rgb[tt].float()/255, person[tt], target_code))
        truth = rgb8(render_style(rgb[tt].float()/255, person[tt], ref_code))
        features = extractor.extract(torch.cat((reference, target)))
        true_codes = torch.cat((ref_code, target_code))
        weights = torch.cat((availability[rr], availability[tt]))
        losses = {}
        for name, model in models.items():
            model.train()
            prediction = model(features['stats'], features['semantic'])
            predicted_reference = fallback_unobserved_roles(prediction[:4], features['person'][:4])
            predicted_target = fallback_unobserved_roles(prediction[4:], features['person'][4:])
            output = compose_styles(target, features['person'][4:], predicted_target, predicted_reference)
            appearance = ((output-truth).abs().mean()+region_l1(output, truth, person[tt]))/2
            parameter = masked_parameter_mse(prediction, true_codes, weights)
            common = availability[rr]*availability[tt]*same.to(device)[:, None]
            consistency = masked_parameter_mse(prediction[:4], prediction[4:], common)
            loss = 10*appearance+args.parameter_weight*parameter+.01*consistency
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite appearance training loss')
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizers[name].step()
            schedulers[name].step()
            losses[name] = {'appearance_mae_codes': float(appearance.detach()*255), 'parameter_mse': float(parameter.detach())}
        if step%100 == 0 or step == 1:
            print(json.dumps({'step': step, 'seconds': time.perf_counter()-started, 'training': losses}), flush=True)
        if step%every == 0 or step == steps:
            gates, rows = {}, {}
            for name, model in models.items():
                gates[name], rows[name] = validate_appearance(model, cache, metadata)
                error = gates[name]['mean_output_mae']
                if name not in best_any or error < best_any[name]['gate']['mean_output_mae']:
                    best_any[name] = {'step': step, 'gate': gates[name]}
                    save_checkpoint(run / f'best-{name}.pt', model, dimensions, name, step, config_sha)
                    (run / f'best-{name}-validation.json').write_text(json.dumps(rows[name], indent=2), encoding='utf-8')
                if gates[name]['passed'] and (name not in best_eligible or error < best_eligible[name]['gate']['mean_output_mae']):
                    best_eligible[name] = {'step': step, 'gate': gates[name]}
                    save_checkpoint(run / f'eligible-{name}.pt', model, dimensions, name, step, config_sha)
                    (run / f'eligible-{name}-validation.json').write_text(json.dumps(rows[name], indent=2), encoding='utf-8')
            history.append({'step': step, 'seconds': time.perf_counter()-started, 'gates': gates})
            report = {'config': config, 'feature_provenance': extractor.provenance, 'completed_steps': step,
                      'validation_history': history, 'best_any': best_any, 'best_eligible': best_eligible,
                      'test_content_loaded': False,
                      'frozen_backbone_trainable_parameters': sum(p.numel() for p in extractor.model.parameters() if p.requires_grad)}
            (run / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'validation_step': step, 'gates': gates}), flush=True)
    if extractor.model.training or any(p.requires_grad for p in extractor.model.parameters()):
        raise ValueError('Frozen feature extractor changed')
    print(json.dumps({'finished': True, 'run_directory': str(run), 'best_eligible': best_eligible, 'best_any': best_any}), flush=True)


if __name__ == '__main__':
    main()
