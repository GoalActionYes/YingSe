"""Exact record extraction for input-fidelity fix and natural-color probes."""
import argparse
import json
from pathlib import Path
from independent_coco import ROOT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-run',required=True);args=parser.parse_args()
    base=ROOT/'outputs/adaptive-reference-20261113/pst50'; hashes={}
    def read(path):
        hashes[str(path.relative_to(ROOT))]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    resolution=read(base/'diagnostic-canonical-resolution-v1/results.json')
    initial=read(base/'development-dense-anchors-v1/results.json')
    native=read(base/'development-dense-anchors-native-v1/results.json')
    cohort=read(base/'development-dense-round2-v1/cohort.json')
    complete=read(base/'development-dense-round2-v1/predictions-complete.json')
    predictions=read(base/'development-dense-round2-v1/predictions.json')
    second=read(base/'development-dense-round2-v1/results.json')
    choice=read(base/'development-global-choice-v1/results.json')
    if len(resolution['cases'])!=5 or len(initial['cases'])!=6 or len(native['cases'])!=6:
        raise ValueError('Incomplete original development')
    if len(second['scores'])!=10 or len(predictions['cases'])!=10 or cohort['development_indices']!=list(range(6,16)):
        raise ValueError('Incomplete new development cohort')
    if complete['predictions_sha256']!=digest(base/'development-dense-round2-v1/predictions.json'):
        raise ValueError('Predictions changed after expert acquisition')
    if len(choice['cases'])!=15: raise ValueError('Incomplete choice diagnostic')
    path=Path(args.native_run).resolve()
    if not path.is_relative_to((ROOT/'outputs/adaptive-reference-20261113/workbench-native-verification').resolve()):
        raise ValueError('Expected own HTTP verification directory')
    http=read(path/'results.json')
    # Do not include arbitrary FFmpeg path-bearing free-text in the Git evidence.
    http['standard']['video_output'].pop('metadata_text',None)
    evidence=dict(scope='Input-fidelity repair plus development probes, not adaptive-product acceptance',
        resolution_diagnostic=resolution,initial_640_comparison=initial,native_comparison=native,
        round2_cohort=cohort,round2_prediction_completion=complete,round2_results=second,
        global_choice=choice,real_http=http,source_result_sha256=hashes,
        builder_sha256=digest(Path(__file__)),remaining_natural_indices=list(range(16,51)),
        dense_candidates_integrated=False,stable_app_algorithm_changed=False,
        current_research_workbench_change='Native decoded inputs replace extra640 analysis thumbnail; preview remains independently reduced')
    out=ROOT/'docs/research/2026-11-13-dense-anchors-and-native-input-evidence.json'
    out.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),bytes=out.stat().st_size,cohort=cohort['development_indices'],
        round2_summary=second['summary'],choice_summary=choice['summary'])),flush=True)


if __name__=='__main__': main()
