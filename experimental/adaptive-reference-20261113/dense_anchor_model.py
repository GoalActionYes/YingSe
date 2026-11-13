"""Pinned lightweight patch descriptors; no image generation or remote code."""
import json
import time
import numpy as np
from PIL import Image
from canonical_model import ROOT,validate_rgb
from independent_coco import digest
from prepare_dense_anchor import REVISION,WEIGHT_SHA


class DenseAnchorModel:
    def __init__(self,device='cuda:0'):
        import torch
        from transformers import Dinov2Model
        directory=ROOT/'.tools/dense-anchor-model'
        manifest=json.loads((directory/'verified-assets.json').read_text(encoding='utf-8'))
        if manifest['revision']!=REVISION or digest(directory/'model.safetensors')!=WEIGHT_SHA:
            raise ValueError('Descriptor provenance mismatch')
        for entry in manifest['files']:
            path=directory/entry['path']
            if path.parent!=directory or digest(path)!=entry['sha256']: raise ValueError('Descriptor asset mismatch')
        self.torch=torch; self.device=torch.device(device)
        self.model=Dinov2Model.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,attn_implementation='sdpa').to(self.device).eval()
        if self.model.config.patch_size!=14 or self.model.config.hidden_size!=384:
            raise ValueError('Unexpected descriptor architecture')
        for p in self.model.parameters(): p.requires_grad_(False)
        self.provenance=dict(repository=manifest['repository'],revision=REVISION,weight_sha256=WEIGHT_SHA,
            preprocessing='Whole RGB, long edge <=448, nearest multiple14 (min14), bilinear; ImageNet mean/std',
            limitation='Correspondence descriptor, not trained grading model; semantic similarity not material or lighting identity')

    def extract(self,rgb):
        validate_rgb(rgb); torch=self.torch
        h,w=rgb.shape[:2]
        scale=min(1.,448/max(h,w))
        width=max(14,round(w*scale/14)*14); height=max(14,round(h*scale/14)*14)
        im=Image.fromarray(rgb).resize((width,height),Image.Resampling.BILINEAR)
        pixel=torch.from_numpy(np.array(im)).permute(2,0,1)[None].to(self.device,dtype=torch.float32)/255
        normalized=(pixel-pixel.new_tensor([.485,.456,.406])[None,:,None,None])/pixel.new_tensor([.229,.224,.225])[None,:,None,None]
        start=time.perf_counter()
        with torch.inference_mode():
            features=self.model(pixel_values=normalized).last_hidden_state[0,1:].float()
            colors=torch.nn.functional.avg_pool2d(pixel,14,14)[0].flatten(1).T
        torch.cuda.synchronize(self.device)
        if features.shape!=(height//14*(width//14),384): raise ValueError('Patch feature/grid mismatch')
        return features,colors,(height//14,width//14),dict(seconds=time.perf_counter()-start,
            working_dimensions=[width,height],original_dimensions=[w,h])
