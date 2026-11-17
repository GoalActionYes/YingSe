"""Verify pinned source/dependencies and import the author pipeline without models."""
import json,sys,os,hashlib,importlib.metadata
from pathlib import Path
import torch,psutil
from independent_coco import ROOT,digest


def main():
    root=ROOT/'.tools/StyleGallery-audit';record=json.loads((ROOT/'docs/research/2026-11-17-stylegallery-assets.json').read_text());counts={}
    for key,folder in [('source',root/'source'),('depth_source',root/'source/pretrained_models/dpv2')]:
        for e in record[key]['files']:
            p=folder/e['path'];assert digest(p)==e['sha256'];data=p.read_bytes();assert hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()==e['git_blob_sha1']
        counts[key]=len(record[key]['files'])
    for e in record['dependencies']:assert digest(root/'wheels'/e['filename'])==e['sha256']
    reuse=ROOT/'.tools/ModFlows-audit';old=json.loads((ROOT/'docs/research/2026-11-17-modflows-evidence.json').read_text())['dependencies']
    for e in old:assert digest(reuse/'wheels'/e['filename'])==e['sha256']
    sys.path[:0]=[str(root/'vendor'),str(reuse/'vendor'),str(root/'source')];os.environ['MPLCONFIGDIR']=str(root/'mpl-cache');os.environ['MPLBACKEND']='Agg'
    from diffusers import StableDiffusionPipeline,DDIMScheduler,UNet2DConditionModel,AutoencoderKL
    from transformers import CLIPTextModel,CLIPTokenizer,AutoModel,AutoImageProcessor
    import pipeline,cv2
    assert Path(pipeline.__file__).resolve()==(root/'source/pipeline.py').resolve()
    assets=[]
    for model in record['models']:
        for e in model['files']:
            path=root/model['target']/e['rfilename'];part=path.with_name(path.name+'.part')
            item=dict(repository=model['repository'],file=e['rfilename'],expected_bytes=e['size'])
            if path.exists():
                assert path.stat().st_size==e['size'];data_hash=hashlib.sha256() if e.get('lfs') else hashlib.sha1()
                if not e.get('lfs'):data_hash.update(('blob '+str(e['size'])+'\0').encode())
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):data_hash.update(chunk)
                assert data_hash.hexdigest()==(e['lfs']['sha256'] if e.get('lfs') else e['blobId']);item['status']='verified'
            else:item.update(status='partial' if part.exists() else 'missing',local_bytes=part.stat().st_size if part.exists() else 0)
            assets.append(item)
    report=dict(scope='Source, isolated dependency and import preflight only; no image inference',source_counts=counts,new_wheels=len(record['dependencies']),reused_wheels=len(old),assets=assets,
        all_model_files_verified=all(e['status']=='verified' for e in assets),imports_passed=True,runtime={n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','accelerate','diffusers']},
        cv2_version=cv2.__version__,physical_ram_available=psutil.virtual_memory().available,cuda_free_total=list(torch.cuda.mem_get_info()),
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['check_stylegallery_preparation.py','fetch_stylegallery_weights.py','stylegallery-screen-protocol.md']},
        asset_registry_sha256=digest(ROOT/'docs/research/2026-11-17-stylegallery-assets.json'),weights_loaded=False,image_inference=False,training=False,product_integrated=False)
    (ROOT/'docs/research/2026-11-17-stylegallery-preparation-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
