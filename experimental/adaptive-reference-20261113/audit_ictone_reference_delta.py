"""Independent file/control reconstruction and difference scoring; no inference."""
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

from ictone_flux_stage import ROOT
from screen_ictone_first import digest
from controlled_styles import stylize
from reconstruct_ictone_lut import author_functions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('.','..')
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    out=base/args.run
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    result,plan=load(out/'results.json'),load(out/'plan.json')
    assert result['finished'] and result['source_sha256']==digest(Path(__file__).with_name('ictone_reference_delta.py'))
    assert digest(out/'plan.json')==result['plan_sha256'] and digest(out/'protocol-frozen.md')==plan['protocol_sha256']
    child_folder=base/plan['child_run']
    child=load(child_folder/'results.json')
    prior=base/'ictone-coco-v1'
    assert child['finished'] and child['checked_parameter_count']==1846 and child['completed_steps']==4
    assert digest(child_folder/'results.json')==result['child_result_sha256']
    assert digest(prior/'results.json')==plan['previous_result_sha256']
    case=plan['previous_case']
    graded=base/case['child_run']
    old=load(graded/'results.json')
    assert old['config']==child['config'] and old['source_sha256']==child['source_sha256']
    assert digest(graded/'content256.png')==digest(child_folder/'content256.png')==digest(out/'content.png')
    for name,sha in child['source_sha256'].items():assert digest(child_folder/name)==sha
    for name,sha in child['output_sha256'].items():assert digest(child_folder/name)==sha
    for name,sha in result['generated_sha256'].items():assert digest(out/name)==sha
    spec=load(out/'input-spec.json')
    assert set(spec)=={'index','content','reference'} and digest(out/'input-spec.json')==plan['spec_sha256']==child['input_spec_sha256']
    def pil(path,resize=True):
        with Image.open(path) as im:
            a=im.convert('RGB')
            return a.resize((256,256),Image.Resampling.BILINEAR) if resize else a.copy()
    def array(im):return np.array(im).astype(float)
    content=pil(out/'content.png')
    raw_reference=pil(coco/'19432-input.png',False)
    assert digest(coco/'19432-input.png')==plan['raw_reference_input_sha256']
    # Recreate only the prescribed known control, never use a truth label to fit.
    control=stylize(np.array(raw_reference),np.zeros((raw_reference.height,raw_reference.width)),'global_contrast')
    ref0=raw_reference.resize((256,256),Image.Resampling.BILINEAR)
    ref1=Image.fromarray(control).resize((256,256),Image.Resampling.BILINEAR)
    assert np.array_equal(np.array(ref0),np.array(pil(out/'reference.png')))
    assert np.array_equal(np.array(ref1),np.array(pil(graded/'reference256.png')))
    target=pil(coco/'9483-input.png',False)
    truth=Image.fromarray(stylize(np.array(target),np.zeros((target.height,target.width)),'global_contrast')).resize((256,256),Image.Resampling.BILINEAR)
    jobs=load(coco/'development/jobs.json')['jobs']
    job=next(x for x in jobs if x['target']==9483 and x['reference']==19432 and x['profile']=='global_contrast')
    assert np.array_equal(np.array(truth),np.array(pil(coco/job['truth_file'])))
    funcs=author_functions()
    lut=np.load(out/'ungraded-reference-lut.npy',allow_pickle=False)
    assert lut.shape==(33,33,33,3) and np.isfinite(lut).all()
    final0=pil(out/'ungraded-reference-lut.png')
    portable=funcs['apply_lut'](np.array(content),lut,device='cpu')
    application_error=int(np.abs(portable.astype(int)-array(final0)).max())
    assert application_error<=1
    final1=pil(prior/(case['index']+'.png'))
    original_plan=load(base/'ictone-coco-inputs-v1/plan.json')
    baseline_path=Path(next(x for x in original_plan['cases'] if x['target']==9483)['baseline_path'])
    baseline0=pil(out/'current13-ungraded-reference.png')
    baseline1=pil(baseline_path)
    c,t=array(content),array(truth)
    expected=t-c
    def stats(im):
        a=array(im)
        return dict(mean_y=float((a@[.299,.587,.114]).mean()),clipped_channel_fraction=float(((a==0)|(a==255)).mean()))
    def score(before,after):
        a,b=array(before),array(after)
        observed=b-a
        den=float(np.sum(expected**2))
        norm=float(np.sqrt(np.sum(observed**2)*den))
        return dict(ungraded_reference_output_change_from_content=float(np.abs(a-c).mean()),
                    graded_reference_error_to_control_gt=float(np.abs(b-t).mean()),
                    observed_increment_mae=float(np.abs(observed).mean()),increment_error_mae=float(np.abs(observed-expected).mean()),
                    increment_projection_on_known_delta=float(np.sum(observed*expected)/den),
                    increment_cosine=float(np.sum(observed*expected)/norm) if norm else None,
                    ungraded_output=stats(before),graded_output=stats(after))
    measured=dict(reference_increment_mae=float(np.abs(array(ref1)-array(ref0)).mean()),
                  known_target_increment_mae=float(np.abs(expected).mean()),input=stats(content),known_control_gt=stats(truth),
                  current13=score(baseline0,baseline1),ictone_lut=score(final0,final1))
    def compare(a,b):
        if isinstance(a,dict):
            assert set(a)==set(b)
            for key in a:compare(a[key],b[key])
        elif a is None or b is None:assert a is b
        else:assert abs(a-b)<=1e-12,(a,b)
    for key,value in measured.items():compare(value,result[key])
    assert abs(measured['ictone_lut']['graded_reference_error_to_control_gt']-case['candidate']['mae'])<=1e-12
    raw0=pil(child_folder/'native512.png')
    audit=dict(finished=True,scope='single-case file and numeric audit, no independent model or natural judgement',
               source_sha256=digest(Path(__file__)),result_sha256=digest(out/'results.json'),
               reconstructed_reference_and_gt_control_exact=True,input_and_weights_config_consistent=True,
               stored_lut_cpu_gpu_max_error=application_error,all_difference_statistics_recomputed=True,
               ungraded_raw_output_change_from_content=float(np.abs(array(raw0)-c).mean()),ungraded_raw_stats=stats(raw0),
               model_rerun_by_auditor=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
