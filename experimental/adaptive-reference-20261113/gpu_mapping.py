"""Optional two-stage global LUT renderer; no per-frame model inference."""
import time
import numpy as np
from canonical_model import validate_rgb


class PreparedMapping:
    def __init__(self,first,second,device=None):
        import torch
        self.torch=torch
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.layers=[]
        for lut in (first,second):
            if (not isinstance(lut,torch.Tensor) or lut.ndim!=5 or lut.shape[:2]!=(1,3) or
                    min(lut.shape[2:])<2 or len(set(lut.shape[2:]))!=1 or not torch.isfinite(lut).all()):
                raise ValueError('Expected finite 1x3xNxNxN LUT')
            self.layers.append(lut.detach().clone().float().to(self.device))

    def apply(self,rgb,strength=1.):
        validate_rgb(rgb)
        if isinstance(strength,(bool,np.bool_)) or not np.isscalar(strength) or not np.isfinite(strength) or not 0<=strength<=1:
            raise ValueError('Expected finite strength in [0,1]')
        if strength==0:return rgb.copy()
        torch=self.torch
        with torch.inference_mode():
            original=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].float().to(self.device)/255
            transformed=original
            for lut in self.layers:
                grid=((transformed-.5)*2).permute(0,2,3,1)[:,None]
                transformed=torch.nn.functional.grid_sample(lut,grid,mode='bilinear',
                            padding_mode='border',align_corners=True)[:,:,0]
            output=(original+float(strength)*(transformed-original)).clamp(0,1)
            return output[0].permute(1,2,0).mul(255).round().byte().cpu().numpy()
