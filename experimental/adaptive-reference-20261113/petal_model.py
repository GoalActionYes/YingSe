"""Isolated official PETAL inference, not an integrated or Jianying engine."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from independent_coco import ROOT,digest
from spatial_render import validate_rgb


def tensor(rgb,device):
    validate_rgb(rgb)
    # Same CPUfloat32 normalization/contiguous CHW as publisher load_image_tensor.
    return torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1).contiguous()[None].to(device)


def rgb8(value):
    if not isinstance(value,torch.Tensor) or value.ndim!=4 or tuple(value.shape[:2])!=(1,3) or min(value.shape[2:])<1 or not value.dtype.is_floating_point or not torch.isfinite(value).all():
        raise ValueError('Expected finite1x3xHxW float output')
    array=value[0].float().permute(1,2,0).cpu().numpy().clip(0,1)
    return np.uint8(array*255+.5)


def verify_dependency(module,wheel):
    origin=getattr(module,'__file__',None)
    if not origin or not Path(origin).resolve().is_relative_to(Path(wheel).resolve()) or getattr(module,'__version__',None)!='0.7.0':
        raise ValueError('Kornia runtime origin/version differs from isolated wheel')


class PetalModel:
    def __init__(self):
        source=ROOT/'.tools/PETAL-source';self.source=source
        m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if not m['ready'] or m['revision']!='162bad662d5a108285bb308699faed2888e3d782' or m['model_revision']!='34eb5176def5f583b72306fea03e0f6b44c6f3bf':raise ValueError('Wrong frozen PETAL assets')
        for row in m['files']+m['weights']:
            p=source/row['path']
            if not p.resolve().is_relative_to(source.resolve()) or digest(p)!=row['sha256']:raise ValueError('PETAL asset mismatch')
        dep=source/m['kornia']['path']
        if digest(dep)!=m['kornia']['sha256']:raise ValueError('Kornia wheel mismatch')
        if 'kornia' in sys.modules:verify_dependency(sys.modules['kornia'],dep)
        for name in ['layers','models_style_encoder','models_lut']:
            existing=sys.modules.get(name)
            if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):raise ValueError('Conflicting external module namespace')
        sys.path[:0]=[str(source),str(dep)]
        import kornia
        verify_dependency(kornia,dep)
        spec=importlib.util.spec_from_file_location('petal_official_infer',source/'infer.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        self.device=torch.device('cuda');self.api=api;self.manifest=m
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            self.style,self.transfer=api.load_models(str(source/'ckpts/style_encoder.pt'),str(source/'ckpts/lut_model.pt'),self.device,torch.float32)
        for model in [self.style,self.transfer]:
            model.eval().requires_grad_(False)
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite restored state')
        self.provenance=dict(source_revision=m['revision'],weights={r['path']:r['sha256'] for r in m['weights']},torch=torch.__version__,gpu=torch.cuda.get_device_name(),kornia_version=kornia.__version__,kornia_runtime_path=str(Path(kornia.__file__).resolve().relative_to(ROOT)),strict_publisher_loader=True,unused_key_omission='vit.mask_token only, publisher loader behavior',precision='publisher no/float32 with its internal RoPE dtype',cudnn_tf32=True,matmul_tf32=False,source_unmodified=True)

    def state_digest(self):
        h=hashlib.sha256()
        for prefix,model in [('style',self.style),('transfer',self.transfer)]:
            for name,value in model.state_dict().items():
                h.update((prefix+'/'+name+':'+str(value.dtype)+':'+str(tuple(value.shape))).encode())
                h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def predict(self,target,reference):
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            result=self.api.inference(self.style,self.transfer,tensor(target,self.device),tensor(reference,self.device),guidance_scale=1.,style_input_size=512)
        image=rgb8(result)
        if image.shape!=target.shape:raise ValueError('PETAL dimensions changed')
        raw=result[0].float().cpu().permute(1,2,0).numpy();torch.cuda.synchronize()
        return image,raw,dict(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
