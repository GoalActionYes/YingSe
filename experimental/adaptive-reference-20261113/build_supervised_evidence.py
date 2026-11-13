"""Collect exact completed research records, without reading any image pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from independent_coco import ROOT,digest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--canonical-run',required=True); args=parser.parse_args()
    base=ROOT/'outputs/adaptive-reference-20261113/supervised-style'
    source_records={}
    def read(path):
        source_records[str(path.relative_to(ROOT))]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    manifest=read(base/'manifest.json')
    splits=Counter(r['split'] for r in manifest['images'])
    if splits!={'train':240,'validation':30,'test':30}: raise ValueError('Dataset split changed')
    parameter=read(base/'training/20261113-065343-f6af40/results.json')
    if parameter['completed_steps']!=1800 or parameter['test_content_loaded']: raise ValueError('Incomplete parameter trial')
    param_appearance=read(base/'training/20261113-065343-f6af40/validation-appearance/results.json')
    appearance={}
    for name,run in [('appearance','20261113-071213-e92eac'),('appearance_warm','20261113-072017-806dee')]:
        report=read(base/f'appearance-training/{run}/results.json')
        if report['completed_steps']!=3600 or report['test_content_loaded']: raise ValueError('Incomplete appearance trial')
        appearance[name]={k:report[k] for k in ('config','completed_steps','best_any','best_eligible','test_content_loaded')}
    pixel=read(base/'pixel-training/20261113-073634-d3962a/results.json')
    if pixel['completed_steps']!=6000 or pixel['test_content_loaded']: raise ValueError('Incomplete pixel trial')
    canpath=Path(args.canonical_run).resolve()
    if not canpath.is_relative_to((base/'canonical-feature-training').resolve()): raise ValueError('Expected own feature trial')
    canonical=read(canpath/'results.json')
    if canonical['completed_steps']!=1800 or canonical['test_content_loaded'] or canonical['config']['stage']!='full':
        raise ValueError('Incomplete feature trial')
    visionbase=ROOT/'outputs/adaptive-reference-20261113/pst50'
    vision=read(visionbase/'development-vision-advisor-v1/results.json')
    categories=read(visionbase/'development-vision-categories-v1/results.json')
    diagnosis=read(visionbase/'diagnostic-vision-advisor-v1/results.json')
    if len(vision['cases'])!=6 or len(categories['cases'])!=7 or len(diagnosis['cases'])!=4:
        raise ValueError('Incomplete vision trial')
    oracle=read(base/'validation-oracle-mask-bound.json')
    if not oracle['model_style_codes_are_oracle'] or oracle['test_content_loaded']:
        raise ValueError('Oracle diagnostic scope changed')
    evidence={
        'scope':'Synthetic validation and first-five natural development probes; no product or final-test acceptance',
        'dataset':dict(counts=dict(splits),manifest_sha256=digest(base/'manifest.json'),
            prior_excluded_count=len(manifest['excluded_prior_ids']),test_status=manifest['test_status'],
            exact_id_and_hash_disjointness='Construction checks exclude prior60 originals; no near-duplicate-series guarantee'),
        'parameter_trial':{k:parameter[k] for k in ('config','completed_steps','best','test_content_loaded')},
        'parameter_appearance_diagnostic':param_appearance['summary'],
        'appearance_trials':appearance,
        'pixel_trial':{k:pixel[k] for k in ('config','completed_steps','best_any','best_eligible','test_content_loaded')},
        'pixel_final_monitor':pixel['history'][-1],
        'oracle_diagnostic':dict(scope=oracle['scope'],known_style_codes_supplied=True,summaries=oracle['summaries']),
        'vision_numeric':vision,'vision_perception_diagnostic':diagnosis,'vision_categorical':categories,
        'canonical_feature_trial':{k:canonical[k] for k in ('config','completed_steps','global_embedding_max_error','best','appearance','parameter_gate','test_content_loaded')},
        'source_result_sha256':source_records,
        'source_sha256':digest(Path(__file__)),
        'product_integrated':False,'final_test_inference_executed':False,
    }
    out=ROOT/'docs/research/2026-11-13-supervised-style-and-vision-evidence.json'
    out.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),bytes=out.stat().st_size,canonical_parameter_gate=canonical['parameter_gate'],
        canonical_appearance={name:value['gate'] for name,value in canonical['appearance'].items()})),flush=True)


if __name__=='__main__': main()
