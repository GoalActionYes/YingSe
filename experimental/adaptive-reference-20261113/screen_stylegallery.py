"""Bounded per-pair native StyleGallery screening with isolated child processes."""
import os,sys,json,hashlib,time,subprocess,traceback,argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.tools/StyleGallery-audit'
BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def checkpoint_key_map(model,keys):
    mapping={key:key for key in keys};original=set(mapping)
    fix=getattr(model,'_fix_state_dict_keys_on_load',None)
    if fix is not None:fix(mapping)
    assert len(mapping)==len(original) and set(mapping.values())==original
    return mapping


def child(position,out):
    # Import numerical/model libraries only in this child's address space.
    config=load(out/'frozen-config.json')
    if config.get('strict_deterministic',False):os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(ASSETS/'mpl-cache'))
    sys.path[:0]=[str(ASSETS/'vendor'),str(ROOT/'.tools/ModFlows-audit/vendor'),str(ASSETS/'source')]
    import torch,numpy as np,psutil
    from PIL import Image
    from safetensors import safe_open
    from diffusers import DDIMScheduler,UNet2DConditionModel,AutoencoderKL
    from transformers import CLIPTextModel,CLIPTokenizer,CLIPImageProcessor
    from torchvision.utils import save_image
    import pipeline,utils
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if config.get('strict_deterministic',False):torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    row=config['rows'][position];folder=out/str(row['index']);folder.mkdir(exist_ok=False);status=dict(index=row['index'],stage='loading',finished=False,activation_checkpoint=config.get('activation_checkpoint',False),strict_deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    def stage(name):
        status.update(stage=name,elapsed_seconds=time.perf_counter()-start);write(folder/'status.json',status);print(name,flush=True)
    start=time.perf_counter();status['initial_resources']=dict(ram_available=psutil.virtual_memory().available,cuda_free_total=list(torch.cuda.mem_get_info()))
    try:
        os.chdir(ASSETS/'source');modeldir=ASSETS/'source/pretrained_models/runwayml_stable-diffusion-v1-5';states={}
        def component(cls,name,filename):
            stage('loading_'+name)
            model=cls.from_pretrained(str(modeldir/name),torch_dtype=torch.float32,device_map={'':'cuda'},low_cpu_mem_usage=True,local_files_only=True,use_safetensors=True)
            actual=model.state_dict()
            with safe_open(str(modeldir/name/filename),framework='pt',device='cuda') as source:
                mapping=checkpoint_key_map(model,source.keys());keys=set(mapping);allowed={'text_model.embeddings.position_ids'} if name=='text_encoder' else set()
                assert set(actual)-keys<=allowed and keys-set(actual)<=allowed,(name,set(actual)^keys)
                verified=0
                for key in keys&set(actual):
                    expected=source.get_tensor(mapping[key]);value=actual[key]
                    assert value.device.type=='cuda' and value.shape==expected.shape and torch.equal(value,expected) and torch.isfinite(value).all(),(name,key)
                    verified+=1
            states[name]=dict(state_tensors_verified=verified,renamed_checkpoint_keys=sum(k!=v for k,v in mapping.items()),parameters=sum(p.numel() for p in model.parameters()),initial_values_exact=True)
            return model
        text=component(CLIPTextModel,'text_encoder','model.safetensors');unet=component(UNet2DConditionModel,'unet','diffusion_pytorch_model.safetensors');vae=component(AutoencoderKL,'vae','diffusion_pytorch_model.safetensors')
        scheduler=DDIMScheduler.from_pretrained(str(modeldir/'scheduler'),local_files_only=True)
        tokenizer=CLIPTokenizer.from_pretrained(str(modeldir/'tokenizer'),local_files_only=True)
        processor=CLIPImageProcessor.from_pretrained(str(modeldir/'feature_extractor'),local_files_only=True)
        pipe=pipeline.StyleGallery(vae=vae,text_encoder=text,tokenizer=tokenizer,unet=unet,scheduler=scheduler,safety_checker=None,feature_extractor=processor,requires_safety_checker=False)
        pipe.extractor=pipe.unet;torch.manual_seed(42);torch.cuda.synchronize();status.update(components=states,load_seconds=time.perf_counter()-start)
        content=str(out/row['inputs']['content']['path']);reference=str(out/row['inputs']['reference']['path'])
        for e in row['inputs'].values():assert digest(out/e['path'])==e['sha256']
        for path in [content,reference]:assert not Path(str(Path(path).with_suffix(''))+'_json').exists()
        torch.cuda.reset_peak_memory_stats();stage('cluster_match');t=time.perf_counter()
        with torch.inference_mode():matches,content_dict,style_dict=pipe.cluster_match(content,[reference],steps=15,content_original_mask=None,use_sam=False,use_depth=True)
        def normal(value):
            if isinstance(value,torch.Tensor):return value.clone()
            if isinstance(value,dict):return {k:normal(v) for k,v in value.items()}
            if isinstance(value,list):return [normal(v) for v in value]
            if isinstance(value,tuple):return tuple(normal(v) for v in value)
            return value
        content_dict,style_dict=normal(content_dict),normal(style_dict);torch.cuda.synchronize();status['cluster_seconds']=time.perf_counter()-t
        status['matched_regions']=sum(v is not None for v in matches.values());status['content_regions']=len(matches)
        status['cluster_peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats()
        stage('style_transfer');t=time.perf_counter();controller=utils.Controller(self_layers=(10,16))
        result=pipe.style_transfer(content_dict=content_dict,style_dict=style_dict,controller=controller,cluster_matches=matches,num_optimize_steps=150,iters=1,lr=.05,c_ratio=.26,enable_gradient_checkpoint=status['activation_checkpoint'])
        torch.cuda.synchronize();status['optimization_seconds']=time.perf_counter()-t
        assert result.shape==(1,3,512,512) and torch.isfinite(result).all() and (result>=0).all() and (result<=1).all()
        save_image(result,folder/'native512.png')
        with Image.open(folder/'native512.png') as im:im.resize((256,256),Image.Resampling.BILINEAR).save(folder/'metric256.png')
        status.update(native_sha256=digest(folder/'native512.png'),metric_sha256=digest(folder/'metric256.png'),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),model_training=False,latent_optimization=True,finished=True)
        stage('complete');write(folder/'result.json',status)
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error).replace(str(ROOT),'<workspace>'),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        stage('failed');write(folder/'failure.json',status);(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


def parent(run,checkpoint=False,bridge=False,resume=None,proof=None,deterministic=False,controlled=False,controlled_baseline=None):
    if controlled:assert deterministic and not bridge and not resume and bool(controlled_baseline)==checkpoint
    if checkpoint and not bridge and not controlled:assert resume and proof,'Checkpoint continuation requires an exact full-image bridge and preserved prior results'
    import numpy as np
    from PIL import Image
    registry=load(ROOT/'docs/research/2026-11-17-stylegallery-assets.json');prep=load(ROOT/'docs/research/2026-11-17-stylegallery-preparation-evidence.json')
    assert prep['all_model_files_verified'] and prep['imports_passed'] and digest(ROOT/'docs/research/2026-11-17-stylegallery-assets.json')==prep['asset_registry_sha256']
    for key,folder in [('source',ASSETS/'source'),('depth_source',ASSETS/'source/pretrained_models/dpv2')]:
        for e in registry[key]['files']:assert digest(folder/e['path'])==e['sha256']
    data=BASE/'cohort-training-coverage-v2';assert digest(data/'cohort.json')=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    assert digest(data/'srgb-v1/manifest.json')=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    rows=load(data/'cohort.json')['groups']['validation'][:1 if bridge or controlled else 8];lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']}
    out=BASE/run;out.mkdir(exist_ok=False);(out/'inputs').mkdir();(out/'frozen-source').mkdir()
    names=['screen_stylegallery.py','stylegallery-screen-protocol.md','stylegallery-runtime-adaptation.md','test_stylegallery_loading.py']
    if checkpoint:
        checked=load(ROOT/'docs/research/2026-11-17-stylegallery-checkpoint-evidence.json');assert checked['passed']
        for n,h in checked['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
        names+=['check_stylegallery_checkpoint.py','stylegallery-checkpoint-protocol.md']
    if deterministic:
        checked=load(ROOT/'docs/research/2026-11-17-stylegallery-deterministic-evidence.json');assert checked['passed'] and checked['deterministic_algorithms'] and not checked['warn_only']
        for n,h in checked['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
        names+=['check_stylegallery_deterministic.py','stylegallery-deterministic-protocol.md','stylegallery-deterministic-image-protocol.md']
    for n in names:(out/'frozen-source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='First8 consumed development style0 rows; necessary-condition screen only',source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},preparation_sha256=digest(ROOT/'docs/research/2026-11-17-stylegallery-preparation-evidence.json'),rows=[],completed=[],finished=False,product_integrated=False,training=False)
    report.update(activation_checkpoint=checkpoint,bridge_only=bridge,expected_count=len(rows),strict_deterministic=deterministic,controlled_single_case=controlled)
    if bridge:assert checkpoint and not resume;report['scope']='First completed case execution bridge only; no new quality selection'
    if controlled:report['scope']='Controlled deterministic single-case execution only; no quality selection'
    def read(relative):
        e=lookup[relative];p=data/e['prepared_path'];assert p.resolve().is_relative_to(BASE.resolve()) and digest(p)==e['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for row in rows:
        identity={}
        for role in ['content','reference']:
            a=read(row[role]);p=out/'inputs'/f"{row['index']}-{role}.png";Image.fromarray(a).save(p);identity[role]=dict(path=str(p.relative_to(out)),sha256=digest(p),pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        report['rows'].append(dict(index=row['index'],inputs=identity))
    if controlled_baseline:
        assert controlled and deterministic and checkpoint and Path(controlled_baseline).name==controlled_baseline
        prior_control=load(BASE/controlled_baseline/'results.json');assert prior_control['finished'] and prior_control['controlled_single_case'] and prior_control['strict_deterministic'] and not prior_control['activation_checkpoint']
        assert prior_control['rows']==report['rows'];report['controlled_baseline_results_sha256']=digest(BASE/controlled_baseline/'results.json')
    if resume:
        assert checkpoint and not bridge and proof and Path(resume).name==resume and Path(proof).name==proof
        bridge_record=load(BASE/proof/'results.json');assert bridge_record['finished'] and bridge_record['bridge']['native_exact'] and bridge_record['bridge']['metric_exact']
        prior=load(BASE/resume/'results.json');assert prior['stop_reason']=='child_failure' and prior['failed_index']==39532
        assert bridge_record['bridge']['prior_results_sha256']==digest(BASE/resume/'results.json')
        oldrows={r['index']:r for r in prior['rows']};newrows={r['index']:r for r in report['rows']}
        for item in prior['completed']:
            index=item['index'];assert oldrows[index]['inputs']==newrows[index]['inputs'];folder=out/str(index);folder.mkdir()
            for filename,key in [('native512.png','native_sha256'),('metric256.png','metric_sha256')]:
                p=BASE/resume/str(index)/filename;assert digest(p)==item[key];(folder/filename).write_bytes(p.read_bytes());assert digest(folder/filename)==item[key]
            copied=dict(item,reused_from=resume);write(folder/'result.json',copied);report['completed'].append(copied)
        report.update(prior_results_sha256=digest(BASE/resume/'results.json'),bridge_results_sha256=digest(BASE/proof/'results.json'))
    write(out/'frozen-config.json',report);start=time.perf_counter();already={r['index'] for r in report['completed']}
    for i,row in enumerate(rows):
        if row['index'] in already:continue
        remaining=2700-(time.perf_counter()-start)
        if remaining<=0:report['stop_reason']='total_budget';break
        logfile=out/f"{row['index']}-child.log"
        with logfile.open('wb') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child','--position',str(i),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1'))
            print(json.dumps(dict(child_pid=process.pid,index=row['index'],position=i)),flush=True)
            try:code=process.wait(timeout=min(600,remaining))
            except subprocess.TimeoutExpired:process.kill();process.wait();report.update(stop_reason='child_timeout',failed_index=row['index']);write(out/'progress.json',report);break
        if code!=0:report.update(stop_reason='child_failure',failed_index=row['index'],child_exit_code=code);write(out/'progress.json',report);break
        item=load(out/str(row['index'])/'result.json');assert item['finished'];report['completed'].append(item);write(out/'progress.json',report);print(json.dumps(dict(completed=len(report['completed']),index=row['index'])),flush=True)
    baseline=load(BASE/'training-coverage-expanded-v2/results.json')['baseline'];byindex={e['index']:e for e in baseline['cases']};measurements=[]
    for item in ([] if bridge or controlled else report['completed']):
        index=item['index'];row=next(r for r in rows if r['index']==index);path=out/str(index)/'metric256.png';assert digest(path)==item['metric_sha256']
        with Image.open(path) as im:pred=np.array(im)
        truth=read(row['gt']);old=BASE/'multiref-control-v1/epoch-6'/f'{index}.png';assert digest(old)==byindex[index]['sha256']
        with Image.open(old) as im:prior=np.array(im)
        old_mae=float(np.abs(prior.astype(float)-truth).mean());assert old_mae==byindex[index]['mae'];mae=float(np.abs(pred.astype(float)-truth).mean())
        measurements.append(dict(index=index,mae=mae,baseline_mae=old_mae,severe=bool(mae-old_mae>5 and mae>old_mae*1.25)))
    report.update(measurements=measurements,severe_regressions=sum(e['severe'] for e in measurements),finished=len(report['completed'])==len(rows),elapsed_seconds=time.perf_counter()-start)
    if bridge and report['finished']:
        prior=BASE/'stylegallery-screen-v3';index=rows[0]['index'];comparisons={}
        for filename,name in [('native512.png','native_exact'),('metric256.png','metric_exact')]:
            with Image.open(out/str(index)/filename) as im:a=np.array(im)
            with Image.open(prior/str(index)/filename) as im:b=np.array(im)
            comparisons[name]=bool(np.array_equal(a,b))
        report['bridge']=dict(comparisons,prior_results_sha256=digest(prior/'results.json'))
    if controlled_baseline and report['finished']:
        differences={};index=rows[0]['index']
        for filename,key in [('native512.png','native_exact'),('metric256.png','metric_exact')]:
            with Image.open(out/str(index)/filename) as im:a=np.array(im)
            with Image.open(BASE/controlled_baseline/str(index)/filename) as im:b=np.array(im)
            differences[key]=bool(np.array_equal(a,b))
        report['controlled_bridge']=dict(differences,passed=all(differences.values()),baseline_results_sha256=report['controlled_baseline_results_sha256'])
    report['full64_severe_gate_impossible']=report['severe_regressions']>=3
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    write(out/'results.json',report);print(json.dumps(dict(finished=report['finished'],completed=len(report['completed']),severe=report['severe_regressions'],stop=report.get('stop_reason'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='stylegallery-screen-v1');parser.add_argument('--child',action='store_true');parser.add_argument('--position',type=int);parser.add_argument('--out',type=Path);parser.add_argument('--checkpoint-enabled',action='store_true');parser.add_argument('--bridge-only',action='store_true');parser.add_argument('--resume-from');parser.add_argument('--bridge-proof');parser.add_argument('--deterministic',action='store_true');parser.add_argument('--controlled-single-case',action='store_true');parser.add_argument('--controlled-baseline');args=parser.parse_args()
    if args.child:child(args.position,args.out)
    else:assert args.run.startswith('stylegallery-screen-') and Path(args.run).name==args.run;parent(args.run,args.checkpoint_enabled,args.bridge_only,args.resume_from,args.bridge_proof,args.deterministic,args.controlled_single_case,args.controlled_baseline)
