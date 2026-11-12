"""GPU renderer for the stable Mapping response, with a bounded table cache."""
from collections import OrderedDict
import numpy as np
from canonical_model import validate_rgb
from reference_color_app.lut import _layers,_strength
from reference_color_app.pipeline import layer_sha256


class PreviewRenderer:
    def __init__(self,device):
        import torch
        self.torch=torch;self.device=torch.device(device);self.tables=OrderedDict()

    def apply(self,rgb,mapping,strength):
        validate_rgb(rgb);t=_strength(strength)
        if t==0:return rgb.copy()
        layers=_layers(mapping);key=layer_sha256(layers)
        torch=self.torch
        if key not in self.tables:
            self.tables[key]=[torch.from_numpy(np.array(layer,copy=True)).permute(3,2,1,0)[None].contiguous().to(self.device) for layer in layers]
            while len(self.tables)>2:self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        first,second=self.tables[key]
        F=np.float32
        area=bool(t>=F(.80001));w=F(np.clip(t/F(.8),0,1));w=F(w*w*F(F(3)-F(2)*w))
        if area:w=F(1)
        mw=F(F(F(1)-t)/F(.2))
        if mw>F(1):mw=F(0)
        with torch.inference_mode():
            x=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].float().to(self.device)/255
            grid=((x-.5)*2).permute(0,2,3,1)[:,None]
            a=torch.nn.functional.grid_sample(first,grid,mode='bilinear',padding_mode='border',align_corners=True)[:,:,0]
            b=torch.nn.functional.grid_sample(second,grid,mode='bilinear',padding_mode='border',align_corners=True)[:,:,0] if area else a
            result=x*float(F(1)-w)+(float(mw)*a+float(F(1)-mw)*b)*float(w)
            return result[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
