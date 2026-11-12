"""Freeze selected numeric evidence without bundling photos, models or sessions."""
import hashlib
import json
from pathlib import Path
from salut_model import ROOT, SOURCE_HASHES, CHECKPOINT_SHA


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113'
    sources = {
        'segmentation': 'independent-coco/development/automatic-masks/results.json',
        'separate_crop': 'independent-coco/development/semantic-candidate/results.json',
        'shared_canonical': 'independent-coco/development/semantic-shared-candidate/results.json',
        'salut_rec709': 'pst50/development-salut/results.json',
        'salut_native_log': 'pst50/development-salut-native-log/results.json',
        'context_ablation': 'pst50/development-context-ablation/results.json',
    }
    records = {key: json.loads((base / path).read_text(encoding='utf-8')) for key, path in sources.items()}
    evidence = {
        'scope': 'Development-only screening, no product integration or general quality claim',
        'source_report_sha256': {key: hashlib.sha256((base / path).read_bytes()).hexdigest() for key, path in sources.items()},
        'segmentation': {'provenance': records['segmentation']['provenance'], 'by_group': records['segmentation']['by_group']},
        'crop_candidates': {key: {field: records[key][field] for field in ('summary', 'by_profile', 'by_group', 'candidate_source_sha256', 'protocol_sha256')}
                            for key in ('separate_crop', 'shared_canonical')},
        'salut': {'upstream_commit': '3e62f9c9b59ea7b30d220ff6ba61047dab898c9a', 'source_hashes': SOURCE_HASHES,
                  'checkpoint_sha256': CHECKPOINT_SHA, 'training_loss_buffer_restored': ['tvmn.weight_c'],
                  'rec709_cases': records['salut_rec709']['cases'], 'native_log_cases': records['salut_native_log']['cases'],
                  'context_ablation_cases': records['context_ablation']['cases'],
                  'native_log_is_not_rgb_adapter': True},
        'tests': {'experimental_unit_checks_passed': 41, 'salut_specific_checks_passed': 9,
                  'compiled_author_extension_end_to_end_comparison': False},
        'evaluation_status': {'coco_held_out_pairs': 24, 'coco_held_out_run': False,
                              'pst50_held_out_pair_indices': list(range(6, 51)), 'pst50_held_out_run': False},
    }
    names = ('salut_model.py', 'salut_interpolation.py', 'check_salut.py', 'check_salut_context.py', 'salut-context-protocol.md')
    evidence['research_source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    destination = ROOT / 'docs/research/2026-11-13-regional-model-screening-evidence.json'
    destination.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(destination.name)


if __name__ == '__main__':
    main()
