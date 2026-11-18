"""Run the fixed MLS support rule on the established ordinary photo controls."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import ROOT, digest
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_mls_auto_anchor_replay import automatic
from probe_mls_auto_foreground import correspond, local_regions
from probe_mls_border_surface import propose
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, decision
from probe_roundtrip_grade_choice import checked_json


@lru_cache(maxsize=80)
def analyze(shape, data):
    return local_regions(np.frombuffer(data, dtype=np.uint8).reshape(shape))


@lru_cache(maxsize=64)
def check_pair(target_shape, target_data, ref_shape, ref_data):
    target = np.frombuffer(target_data, dtype=np.uint8).reshape(target_shape)
    reference = np.frombuffer(ref_data, dtype=np.uint8).reshape(ref_shape)
    photo, detail = automatic(target, reference)
    pairs = []
    if detail['anchor']:
        pairs = correspond(analyze(target_shape,target_data), analyze(ref_shape,ref_data),
                           np.array(detail['offset']))
    witnesses = sum(pair['supports'] for pair in pairs)
    return photo, dict(anchor=detail['anchor'], correspondences=len(pairs), witnesses=witnesses,
                       applied=witnesses>=2, same_image=detail['identity'])


class Probe:
    def __init__(self, preserve_baseline):
        self.preserve_baseline = preserve_baseline

    def measure(self, target, reference, baseline):
        rendered, detail = check_pair(target.shape,target.tobytes(),reference.shape,reference.tobytes())
        if detail['same_image']:
            return target.copy(), detail
        if not detail['applied']:
            return (baseline if self.preserve_baseline else target).copy(), detail
        if not self.preserve_baseline:
            return rendered.copy(), detail
        thumbnail, _ = propose(target)
        background = np.asarray(Image.fromarray(thumbnail).resize((target.shape[1],target.shape[0]),
                                                                  Image.Resampling.NEAREST))
        area = target_regions(target,background)['neutral_proxy'] & ~background
        result = baseline.copy()
        result[area] = rendered[area]
        return result, detail


def summarize(background, local, real):
    control = decision(background)
    local_count = whole_count = empty_count = 0
    for row in local.values():
        checks = row['gates']['checks']
        local_count += row['gates']['positive'] and all(
            value is not False for key,value in checks.items() if key.startswith('local'))
        whole_count += checks['whole_response']
        empty_count += checks['no_person'] is True
    real_pass = any(row['gate'] for row in real.values()) and all(
        measures['methods']['candidate']['mae']<=measures['methods']['current13']['mae']+1
        for row in real.values() for measures in row['metrics'].values())
    applied = sum(row['applied'] for scene in background for row in scene['variants'].values())
    applied += sum(row['applied'] for scene in local.values() for row in scene['output'].values())
    applied += sum(row['applied'] for row in real.values())
    return dict(background=control, local_pass=local_count, whole_pass=whole_count, no_person_pass=empty_count,
                real_pass=real_pass, real_cases={name:row['gate'] for name,row in real.items()}, applied=applied,
                passed=bool(control['proceed'] and local_count>=3 and whole_count>=4 and empty_count==2 and real_pass))


def run(out):
    docs = ROOT/'docs/research'
    toggle = checked_json(docs/'2026-11-18-local-edit-toggle-evidence.json',PARENT_REPORT)
    parent = checked_json(PARENT/'results.json',PARENT_SHA)
    assert toggle['parent_sha256'] == PARENT_SHA and parent['strength']==.8
    psbench = docs/'2026-11-18-psbench-action-pair-evidence.json'
    assert digest(psbench)==PSBENCH_SHA
    out.mkdir()
    cases = {}
    synthetic = np.full((32,32,3),127,np.uint8)
    for method, preserves in (('raw_replay',False), ('baseline_fallback',True)):
        probe = Probe(preserves)
        identity, state = probe.measure(synthetic,synthetic.copy(),255-synthetic)
        assert np.array_equal(identity,synthetic) and state['same_image']
        path = out/method
        path.mkdir()
        background = background_controls(path,parent,probe.measure)
        local = local_controls(probe,path,toggle,{item['pair']:item for item in parent['cases']})
        real = authored(path,BASE/'psbench-shared-local-v1',psbench,probe.measure)
        cases[method] = dict(background=background,local=local,real=real,summary=summarize(background,local,real))
    data = dict(methods=cases, background_sha256=PARENT_SHA, toggle_sha256=PARENT_REPORT,
                psbench_sha256=PSBENCH_SHA, source_sha256=digest(Path(__file__)),
                matcher_source_sha256=digest(Path(__file__).with_name('probe_mls_auto_foreground.py')))
    (out/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({method:row['summary'] for method,row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
