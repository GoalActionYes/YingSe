"""Pretrained grading-feature ablation under unchanged synthetic controls."""
import argparse
import json
import time
import uuid
from pathlib import Path
import torch
from independent_coco import ROOT,digest
from canonical_style_features import CanonicalStyleFeatures
from train_style_probe import load_images,validation_features,validate
from style_probe import StyleRegressor,render_style,draw_style_codes,masked_parameter_mse,PARAM_SCALE
from appearance_data import build_validation,validate_appearance


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    steps,every=(8,4) if args.smoke else (1800,150)
    data=ROOT/'outputs/adaptive-reference-20261113/supervised-style'
    manifest=data/'manifest.json'
    run=data/'canonical-feature-training'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    names=('train_canonical_style_probe.py','canonical_style_features.py','canonical-feature-protocol.md',
           'canonical_model.py','style_probe.py','style_features.py','person_segmenter.py','train_style_probe.py',
           'appearance_data.py','appearance_metrics.py','style_composition.py')
    frozen=run/'frozen-source'; frozen.mkdir()
    hashes={}
    for name in names:
        source=Path(__file__).with_name(name); (frozen/name).write_bytes(source.read_bytes()); hashes[name]=digest(source)
    config=dict(stage='smoke' if args.smoke else 'full',scope='Synthetic grading-feature probe, not product acceptance',
        steps=steps,validation_every=every,batch=8,style_pairs_per_batch=4,learning_rate=.001,
        weight_decay=.0001,consistency_weight=.1,seed=20261113,validation_seed=2026111301,
        appearance_validation_seed=2026111307,parameter_scale=list(PARAM_SCALE),
        source_sha256=hashes,manifest_sha256=digest(manifest),test_content_loaded=False)
    config_path=run/'config.json'; config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    torch.manual_seed(20261113); torch.set_num_threads(4); device=torch.device('cuda:0')
    rgb,person,available,records=load_images(manifest,'train',device)
    vrgb,vperson,vavailable,vrecords=load_images(manifest,'validation',device)
    if {r['id'] for r in records}&{r['id'] for r in vrecords}: raise ValueError('Split overlap')
    extractor=CanonicalStyleFeatures(str(device))
    cached=validation_features(extractor,vrgb,vperson,vavailable)
    with torch.no_grad():
        images=vrgb[:4].float()/255
        actual=extractor.extract(images)['semantic'][:,:64]
        expected=extractor.encoder(images)
        parity=float((actual-expected).abs().max())
        if parity>1e-6: raise ValueError('Global feature parity failed')
    dimensions=cached['stats'].shape[1],cached['semantic'].shape[1]
    models,optimizers,best={},{},{}
    for name,semantic in (('color_statistics',False),('grading_features',True)):
        torch.manual_seed(20261113)
        model=StyleRegressor(*dimensions,use_semantic=semantic).to(device)
        models[name]=model
        optimizers[name]=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best[name]={'loss':float('inf')}
    generator=torch.Generator().manual_seed(20261113)
    history=[]; started=time.perf_counter()
    report=dict(config=config,feature_provenance=extractor.provenance,global_embedding_max_error=parity,
        completed_steps=0,best=best,validation_history=history,test_content_loaded=False)
    result_path=run/'results.json'
    print(json.dumps(dict(run_directory=str(run),stage=config['stage'],global_embedding_max_error=parity)),flush=True)
    for step in range(1,steps+1):
        first=torch.randint(len(rgb),(4,),generator=generator)
        second=(first+torch.randint(1,len(rgb),(4,),generator=generator))%len(rgb)
        ii=torch.stack((first,second),dim=1).flatten().to(device)
        codes=draw_style_codes(4,generator).repeat_interleave(2,dim=0).to(device)
        styled=render_style(rgb[ii].float()/255,person[ii],codes)
        features=extractor.extract(styled); weights=available[ii]
        losses={}
        for name,model in models.items():
            model.train(); prediction=model(features['stats'],features['semantic'])
            supervised=masked_parameter_mse(prediction,codes,weights)
            common=weights[::2]*weights[1::2]
            loss=supervised+.1*masked_parameter_mse(prediction[::2],prediction[1::2],common)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            optimizers[name].zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizers[name].step(); losses[name]=float(supervised.detach())
        if step%every==0 or step==steps:
            results={name:validate(model,cached) for name,model in models.items()}
            history.append(dict(step=step,seconds=time.perf_counter()-started,training_mse=losses,validation=results))
            for name,result in results.items():
                if result['normalized_parameter_mse']<best[name]['loss']:
                    best[name]=dict(loss=result['normalized_parameter_mse'],step=step,validation=result)
                    torch.save(dict(model={k:v.detach().cpu() for k,v in models[name].state_dict().items()},
                        stats_dim=dimensions[0],semantic_dim=dimensions[1],use_semantic=name=='grading_features',
                        step=step,config_sha256=digest(config_path)),run/f'best-{name}.pt')
            report.update(completed_steps=step,best=best)
            result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(dict(validation_step=step,results=results)),flush=True)
    # This diagnostic has a different fixed seed/protocol from parameter validation.
    # Neither checkpoint selection nor weight updates use appearance scores.
    cache,metadata=build_validation(extractor,vrgb,vperson,vavailable,vrecords)
    appearance={}
    for name,model in models.items():
        path=run/f'best-{name}.pt'
        payload=torch.load(path,weights_only=True,map_location='cpu')
        if payload['config_sha256']!=digest(config_path) or payload['step']!=best[name]['step']:
            raise ValueError('Checkpoint lineage mismatch')
        model.load_state_dict(payload['model'],strict=True)
        gate,rows=validate_appearance(model,cache,metadata)
        appearance[name]=dict(gate=gate,checkpoint_sha256=digest(path))
        (run/f'appearance-{name}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if any(p.requires_grad for m in (extractor.model,extractor.encoder) for p in m.parameters()):
        raise ValueError('Frozen feature weights became trainable')
    report.update(appearance=appearance,parameter_gate={name:value['validation']['relative_mse_to_zero']<=.8 for name,value in best.items()})
    result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,run_directory=str(run),parameter_gate=report['parameter_gate'],appearance=appearance)),flush=True)


if __name__=='__main__': main()
