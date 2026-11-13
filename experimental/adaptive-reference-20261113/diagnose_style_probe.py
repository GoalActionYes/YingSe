"""Validation-only appearance diagnostic after parameter-identifiability gate failure.

Never reads final test records. Quantizes graded references like real RGB8 input.
Human target masks generate/score truth and one labeled oracle-mask diagnostic;
the actual candidate renderer receives automatic RGB-derived target probabilities.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT, digest
from train_style_probe import load_images, fixed_validation_codes
from style_features import StyleFeatures
from style_probe import StyleRegressor, render_style, masked_parameter_mse


def rgb8(tensor):
    return tensor.mul(255).round().clamp(0, 255).byte().float()/255


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    data = ROOT / 'outputs/adaptive-reference-20261113/supervised-style'
    run = Path(args.run).resolve()
    if not run.is_relative_to((data / 'training').resolve()):
        raise ValueError('Expected own supervised-style training directory')
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    history = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    if config['stage'] != 'full' or history['completed_steps'] != 1800:
        raise ValueError('Expected completed full run')
    for name, expected in config['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Training source changed before diagnostic')
    manifest_path = data / 'manifest.json'
    if digest(manifest_path) != config['manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    device = torch.device('cuda:0')
    rgb, person, availability, records = load_images(manifest_path, 'validation', device)
    original = rgb.float()/255
    extractor = StyleFeatures(str(device))
    auto_person = torch.cat([extractor.extract(original[i:i+8])['person'] for i in range(0, len(rgb), 8)])
    controls = fixed_validation_codes().to(device)
    pairs = []
    for group in ('single', 'multiple', 'none'):
        members = [i for i, record in enumerate(records) if record['group'] == group]
        pairs.extend(zip(members[::2], members[1::2]))
    out = run / 'validation-appearance'
    out.mkdir(exist_ok=False)
    methods = {}
    weights_sha = {}
    for name in ('color_statistics', 'semantic_features'):
        path = run / f'best-{name}.pt'
        payload = torch.load(path, weights_only=True, map_location='cpu')
        if payload['config_sha256'] != digest(run / 'config.json'):
            raise ValueError('Checkpoint/config mismatch')
        model = StyleRegressor(payload['stats_dim'], payload['semantic_dim'], payload['use_semantic']).to(device)
        model.load_state_dict(payload['model'], strict=True)
        methods[name] = model.eval()
        weights_sha[name] = digest(path)
    rows = []
    for style_index, control in enumerate(controls):
        codes = control[None].expand(len(rgb), -1, -1)
        reference = rgb8(render_style(original, person, codes))
        features = [extractor.extract(reference[i:i+8]) for i in range(0, len(rgb), 8)]
        stats, semantic = torch.cat([x['stats'] for x in features]), torch.cat([x['semantic'] for x in features])
        predictions = {name: model(stats, semantic) for name, model in methods.items()}
        for target_index, reference_index in pairs:
            target = original[target_index:target_index+1]
            truth_mask = person[target_index:target_index+1]
            mask = auto_person[target_index:target_index+1]
            true_code = control[None]
            truth = rgb8(render_style(target, truth_mask, true_code))
            row = {'target': records[target_index]['id'], 'reference': records[reference_index]['id'],
                   'group': records[target_index]['group'], 'style_index': style_index,
                   'identity_mae': float((target-truth).abs().mean()*255), 'methods': {}}
            for name, prediction in predictions.items():
                predicted = prediction[reference_index:reference_index+1]
                result = rgb8(render_style(target, mask, predicted))
                oracle_mask_result = rgb8(render_style(target, truth_mask, predicted))
                row['methods'][name] = {'auto_mask_mae': float((result-truth).abs().mean()*255),
                                       'oracle_mask_diagnostic_mae': float((oracle_mask_result-truth).abs().mean()*255),
                                       'parameter_mse': float(masked_parameter_mse(predicted, true_code, availability[reference_index:reference_index+1]))}
                if style_index in (0, 5):
                    picture = result[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
                    Image.fromarray(picture).save(out / f'{records[target_index]["id"]}-style{style_index}-{name}.png')
            rows.append(row)
    summary = {}
    for name in methods:
        baseline = np.array([r['identity_mae'] for r in rows])
        actual = np.array([r['methods'][name]['auto_mask_mae'] for r in rows])
        identity = [r['methods'][name]['auto_mask_mae'] for r in rows if r['style_index'] < 2]
        summary[name] = {'mean_identity_mae': float(baseline.mean()), 'mean_candidate_mae': float(actual.mean()),
                         'median_identity_mae': float(np.median(baseline)), 'median_candidate_mae': float(np.median(actual)),
                         'mean_false_adjustment_on_identity_controls': float(np.mean(identity)),
                         'improved_cases_over_identity': int((actual < baseline-.1).sum())}
    report = {'scope': 'Validation-only cross-content diagnostic, not final test or product acceptance',
              'training_config_sha256': digest(run / 'config.json'), 'weights_sha256': weights_sha,
              'diagnostic_source_sha256': digest(Path(__file__)), 'summary': summary, 'cases': rows,
              'test_content_loaded': False}
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'summary': summary, 'test_content_loaded': False}), flush=True)


if __name__ == '__main__':
    main()
