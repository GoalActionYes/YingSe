"""Research-only native-size bilateral-grid renderer with bounded GPU bands.

Uses the fixed academically fine-tuned Hist2Style model. All transforms keep
pixel positions; a spatial grid cannot be faithfully exported as one RGB CUBE.
"""
import numbers
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output
from reference_color_app.adjustments import transform
from reference_color_app.lut import _strength

CHECKPOINT_SHA='d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'


def checkpoint_for(profile):
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    if profile=='app8':return base/'training-v1/epoch-8.pth',CHECKPOINT_SHA
    if profile=='multiref-v1':return base/'multiref-control-v1/epoch-6.pth','d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    raise ValueError('Unknown spatial checkpoint profile')


def validate_rgb(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('Expected nonempty RGB8 image')


def finish_spatial(rgb,graded,strength,adjustments=None):
    validate_rgb(rgb);validate_rgb(graded)
    if rgb.shape!=graded.shape:raise ValueError('Grade dimensions changed')
    strength=float(_strength(strength))
    if strength==0:value=rgb.astype(np.float32)/255
    elif strength==1:value=graded.astype(np.float32)/255
    else:value=rgb.astype(np.float32)/255*(1-strength)+graded.astype(np.float32)/255*strength
    return np.rint(transform(value,adjustments)*255).astype(np.uint8)


def coordinates(width,height,start,stop,z):
    if any(type(v)!=int for v in [width,height,start,stop]) or width<1 or height<1 or not 0<=start<stop<=height or tuple(z.shape)!=(1,stop-start,width):
        raise ValueError('Invalid full-image band coordinates')
    x=(2*torch.arange(width,device=z.device,dtype=torch.float32)+1)/width-1
    y=(2*torch.arange(start,stop,device=z.device,dtype=torch.float32)+1)/height-1
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    return torch.stack([xx[None],yy[None],(z-.5)*2],-1).unsqueeze(1).clamp(-1,1)


class SpatialEngine:
    def __init__(self,profile='app8'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        checkpoint,self.checkpoint_sha=checkpoint_for(profile)
        if digest(checkpoint)!=self.checkpoint_sha:raise ValueError('Frozen spatial checkpoint changed')
        self.model,self.api,self.author=author_model()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
        self.model.load_state_dict(state,strict=True);self.model.eval().requires_grad_(False)
        self.device=next(self.model.parameters()).device

    def tensor(self,rgb):
        return torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(self.device,dtype=torch.float32)/255

    def analysis(self,rgb):
        validate_rgb(rgb)
        small=np.array(Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR))
        return self.tensor(small)

    @torch.inference_mode()
    def prepare(self,target,reference):
        start=time.perf_counter();c,s=self.analysis(target),self.analysis(reference)
        hc=self.api['batch_histogram_1D'](self.api['srgb_to_ycbcr'](c));hs=self.api['batch_histogram_1D'](self.api['srgb_to_ycbcr'](s))
        low=self.api['TF'].resize(c,(64,64),antialias=True)
        grid=self.api['lowpass'](self.model.apply_grid(low,torch.cat([hc,hs],1)))
        finite_output(grid,(1,13,8,16,16));torch.cuda.synchronize(self.device)
        return PreparedSpatial(self,grid.detach().clone(),dict(analysis_seconds=time.perf_counter()-start,
            analysis_size=[256,256],original_dimensions=[target.shape[1],target.shape[0]],checkpoint_sha256=self.checkpoint_sha))


class PreparedSpatial:
    def __init__(self,engine,grid,diagnostics):self.engine,self.grid,self.diagnostics=engine,grid,diagnostics

    @torch.inference_mode()
    def float_output(self,rgb,chunk_rows=128):
        validate_rgb(rgb)
        if type(chunk_rows)!=int or chunk_rows<1:raise ValueError('Positive integer band height required')
        model=self.engine.model;height,width=rgb.shape[:2];output=np.empty(rgb.shape,np.float32)
        weights=F.softmax(model.sampler.L_weights,dim=1)
        for start in range(0,height,chunk_rows):
            stop=min(start+chunk_rows,height);c=self.engine.tensor(rgb[start:stop])
            guide=model.sampler.pre_guide(c);z=(guide*weights).sum(1)
            finite_output(z,(1,stop-start,width))
            coords=coordinates(width,height,start,stop,z)
            coeff=F.grid_sample(self.grid,coords,mode='bilinear',align_corners=False,padding_mode='border').squeeze(2)
            raw=finite_output(model.apply_coeffs(coeff,c),c.shape)
            output[start:stop]=raw[0].cpu().permute(1,2,0).numpy()
        return output

    def apply(self,rgb,strength=1.,chunk_rows=128):
        validate_rgb(rgb)
        if isinstance(strength,(bool,np.bool_)) or not isinstance(strength,numbers.Real) or not np.isfinite(strength) or not 0<=strength<=1:
            raise ValueError('Strength must be a finite number within 0..1')
        if strength==0:return rgb.copy()
        graded=self.float_output(rgb,chunk_rows)
        blended=rgb.astype(np.float32)/255*(1-float(strength))+graded*float(strength)
        return np.rint(np.clip(blended,0,1)*255).astype(np.uint8)
