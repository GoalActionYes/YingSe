"""Prepare one fixed reference ablation, then measure after its child exits."""
import argparse
import json
import time
from pathlib import Path

from ictone_flux_stage import ROOT
from screen_ictone_first import digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--measure',action='store_true')
    args=p.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('.','..')
    import numpy as np
    from PIL import Image,ImageDraw
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    out=base/args.run
    previous=base/'ictone-coco-v1'
    old=json.loads((previous/'results.json').read_text())
    assert old['finished'] and json.loads((previous/'audit.json').read_text())['finished']
    case=next(x for x in old['completed'] if x['target']==9483)
    graded=base/case['child_run']
    def read(path):
        with Image.open(path) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    if not args.measure:
        out.mkdir(exist_ok=False)
        manifest=json.loads((coco/'manifest.json').read_text())
        raw=next(x for x in manifest['images'] if x['id']==19432)
        assert digest(coco/'19432-input.png')==raw['input_sha256']
        (out/'content.png').write_bytes((graded/'content256.png').read_bytes())
        read(coco/'19432-input.png').save(out/'reference.png')
        spec=dict(index='reference-delta-9483',content=dict(path=str(out/'content.png'),sha256=digest(out/'content.png')),
                  reference=dict(path=str(out/'reference.png'),sha256=digest(out/'reference.png')))
        (out/'input-spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        protocol=ROOT/'docs/research/2026-11-17-ictone-reference-delta.md'
        (out/'protocol-frozen.md').write_bytes(protocol.read_bytes())
        plan=dict(finished=True,child_run=args.run+'-native',spec_sha256=digest(out/'input-spec.json'),
                  previous_result_sha256=digest(previous/'results.json'),previous_case=case,
                  raw_reference_input_sha256=raw['input_sha256'],protocol_sha256=digest(protocol),
                  source_sha256=digest(Path(__file__)))
        (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        print(json.dumps(dict(prepared=True,child_run=plan['child_run'],spec=str(out/'input-spec.json'))))
        return
    import torch
    from reconstruct_ictone_lut import author_functions
    from train_triplet_pilot import author_model
    from appearance_replication import initialize_replication_model
    from appearance_pairwise import appearance_forward
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    plan=json.loads((out/'plan.json').read_text())
    assert plan['source_sha256']==digest(Path(__file__))
    assert digest(previous/'results.json')==plan['previous_result_sha256']
    child_folder=base/plan['child_run']
    child=json.loads((child_folder/'results.json').read_text())
    assert child['finished'] and child['checked_parameter_count']==1846
    assert child['input_spec_sha256']==plan['spec_sha256']==digest(out/'input-spec.json')
    for role in ['content','reference']:
        assert digest(child_folder/(role+'256.png'))==digest(out/(role+'.png'))
    funcs=author_functions()
    content=read(out/'content.png')
    ref0=read(out/'reference.png')
    ref1=read(graded/'reference256.png')
    raw0=read(child_folder/'native512.png')
    lut=funcs['estimate_lut'](np.array(content).reshape(-1,3),np.array(raw0).reshape(-1,3),size=33,device='cuda')
    assert np.isfinite(lut).all()
    final0=Image.fromarray(funcs['apply_lut'](np.array(content),lut,device='cuda'))
    final0.save(out/'ungraded-reference-lut.png')
    np.save(out/'ungraded-reference-lut.npy',lut)
    final1=read(previous/(case['index']+'.png'))
    assert digest(previous/(case['index']+'.png'))==case['output_sha256']
    # GT is opened only after the new prediction and LUT are saved.
    jobs=json.loads((coco/'development/jobs.json').read_text())['jobs']
    job=next(x for x in jobs if x['target']==9483 and x['reference']==19432 and x['profile']=='global_contrast')
    truth=read(coco/job['truth_file'])
    checkpoint=base/'multiref-control-v1/epoch-6.pth'
    assert digest(checkpoint)=='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    model,api,_=author_model()
    initialize_replication_model(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval().requires_grad_(False)
    def predict(ref):
        tensor=lambda im:(torch.from_numpy(np.array(im).copy()).permute(2,0,1)[None].float()/255).cuda()
        with torch.inference_mode():raw=appearance_forward(model,api,tensor(content),tensor(ref))[0]
        return Image.fromarray(np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8))
    baseline0,baseline1=predict(ref0),predict(ref1)
    original_plan=json.loads((base/'ictone-coco-inputs-v1/plan.json').read_text())
    baseline_path=Path(next(x for x in original_plan['cases'] if x['target']==9483)['baseline_path'])
    assert np.array_equal(np.array(baseline1),np.array(read(baseline_path)))
    baseline0.save(out/'current13-ungraded-reference.png')
    def array(im):return np.array(im).astype(float)
    c,t=array(content),array(truth)
    expected=t-c
    def stats(im):
        a=array(im)
        return dict(mean_y=float((a@[.299,.587,.114]).mean()),
                    clipped_channel_fraction=float(((a==0)|(a==255)).mean()))
    def compare(before,after):
        a,b=array(before),array(after)
        observed=b-a
        den=float(np.sum(expected*expected))
        assert den>0
        norm=float(np.sqrt(np.sum(observed*observed)*den))
        return dict(ungraded_reference_output_change_from_content=float(np.abs(a-c).mean()),
                    graded_reference_error_to_control_gt=float(np.abs(b-t).mean()),
                    observed_increment_mae=float(np.abs(observed).mean()),
                    increment_error_mae=float(np.abs(observed-expected).mean()),
                    increment_projection_on_known_delta=float(np.sum(observed*expected)/den),
                    increment_cosine=float(np.sum(observed*expected)/norm) if norm else None,
                    ungraded_output=stats(before),graded_output=stats(after))
    elapsed=time.monotonic()-started
    assert elapsed<=120,elapsed
    result=dict(finished=True,scope='one consumed reference intervention, not causal attribution percentage or quality acceptance',
                source_sha256=digest(Path(__file__)),plan_sha256=digest(out/'plan.json'),
                child_result_sha256=digest(child_folder/'results.json'),
                fixed_inputs_verified=True,baseline_graded_pixels_exact=True,
                reference_increment_mae=float(np.abs(array(ref1)-array(ref0)).mean()),
                known_target_increment_mae=float(np.abs(expected).mean()),
                input=stats(content),known_control_gt=stats(truth),
                current13=compare(baseline0,baseline1),ictone_lut=compare(final0,final1),
                wall_measurement_seconds=elapsed,image_pipeline_seconds=child['image_pipeline_seconds'],
                generated_sha256={n:digest(out/n) for n in ['ungraded-reference-lut.png','ungraded-reference-lut.npy','current13-ungraded-reference.png']})
    panel=Image.new('RGB',(1536,564),'white')
    draw=ImageDraw.Draw(panel)
    rows=[[('Content',content),('Reference original',ref0),('Reference contrast',ref1),('GT control',truth),
            ('C13 original ref',baseline0),('C13 contrast ref',baseline1)],
          [('Content',content),('Reference original',ref0),('Reference contrast',ref1),('GT control',truth),
            ('ICTone original ref',final0),('ICTone contrast ref',final1)]]
    for j,row in enumerate(rows):
        for i,(label,im) in enumerate(row):
            panel.paste(im,(256*i,282*j+26));draw.text((256*i+4,282*j+7),label,fill='black')
    panel.save(out/'diagnostic-panel.png')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
