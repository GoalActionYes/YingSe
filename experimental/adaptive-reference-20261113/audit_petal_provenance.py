"""Postscreen dependency-origin audit; does not rerun or change predictions."""
import importlib,json,sys
from pathlib import Path
from independent_coco import ROOT,digest
from petal_model import verify_dependency


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/petal-20261115-v1'
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    source=ROOT/'.tools/PETAL-source';manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'));wheel=source/manifest['kornia']['path']
    if digest(wheel)!=manifest['kornia']['sha256']:raise ValueError('Wheel changed')
    sys.path.insert(0,str(wheel));kornia=importlib.import_module('kornia');verify_dependency(kornia,wheel)
    changes={name:dict(frozen_sha256=sha,current_sha256=digest(Path(__file__).with_name(name))) for name,sha in report['source_sha256'].items() if digest(Path(__file__).with_name(name))!=sha}
    if set(changes)!={'petal_model.py','test_petal_model.py','prepare_petal.py'}:raise ValueError('Unexpected changes after frozen screen')
    for name,sha in report['source_sha256'].items():
        if digest(out/'frozen-source'/name)!=sha:raise ValueError('Frozen evidence changed')
    audit=dict(scope='Postscreen fresh-process dependency-origin check; not retrospective proof of original process importpath, not new inference',passed=True,results_sha256=digest(out/'results.json'),verifier_sha256=digest(Path(__file__)),kornia=dict(version=kornia.__version__,runtime_path=str(Path(kornia.__file__).resolve().relative_to(ROOT)),wheel_sha256=digest(wheel),installed=False),postscreen_source_changes=changes,change_reason='Fail-closed runtime wheel origin/version validation and its test; acquisition pins the same immutable source revision instead of resolving master; publisherAPI/image math unchanged; no prediction rerun',original_model_state_before=report['model_state_before'],original_model_state_after=report['model_state_after'])
    if report['model_state_before']!=report['model_state_after']:raise ValueError('Original model state changed')
    (out/'postscreen-provenance.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,version=kornia.__version__,source_changes=list(changes)),indent=2))


if __name__=='__main__':main()
