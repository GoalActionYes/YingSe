"""Serialized model preparation and float LUT compilation for the research UI."""
import threading
import traceback
import numpy as np
from reference_color_app.contracts import Mapping
from learned_store import identity_cube


def compile_luts(first,second,diagnostics):
    import torch
    rgb=identity_cube()
    with torch.inference_mode():
        x=torch.from_numpy(rgb.reshape(512,512,3)).permute(2,0,1)[None].to(first.device)
        for lut in (first,second):
            grid=((x-.5)*2).permute(0,2,3,1)[:,None]
            x=torch.nn.functional.grid_sample(lut,grid,mode='bilinear',padding_mode='border',align_corners=True)[:,:,0]
        final=x[0].permute(1,2,0).clamp(0,1).reshape(64,64,64,3).cpu().numpy().copy()
    return Mapping(np.stack([.2*rgb+.8*final,final]).astype(np.float32),dict(diagnostics,
                   compilation='Two float model LUTs sampled into one global 64-cube; interpolation approximation'))


class ModelProvider:
    def __init__(self):
        self.lock=threading.RLock();self.model=None;self.state='loading';self.renderer=None
        threading.Thread(target=self._load,daemon=True,name='adaptive-model-load').start()

    def _load(self):
        try:
            from canonical_model import CanonicalModel
            model=CanonicalModel()
            # First CUDA execution has extra setup cost. Keep it outside UI requests.
            probe=np.full((32,32,3),128,np.uint8)
            model.predict(probe,probe)
            with self.lock:self.model=model;self.state='ready'
        except Exception:
            traceback.print_exc()
            with self.lock:self.state='error'

    def status(self):
        with self.lock:return dict(state=self.state,message={'loading':'模型准备中，首次启动可能较慢',
                  'ready':'模型已就绪','error':'模型准备失败，可使用现有算法；详情见实验日志'}[self.state])

    def compile(self,target,reference):
        with self.lock:
            if self.state!='ready':raise ValueError(self.status()['message'])
            _,timing=self.model.predict(target,reference)
            return compile_luts(*self.model.last_luts,dict(engine='model',provenance=self.model.provenance,
                                analysis_dimensions=[target.shape[1],target.shape[0]],timing=timing))

    def preview(self,rgb,mapping,strength):
        with self.lock:
            if self.state=='ready' and self.model.device.type=='cuda':
                if self.renderer is None:
                    from gpu_preview import PreviewRenderer
                    self.renderer=PreviewRenderer(self.model.device)
                return self.renderer.apply(rgb,mapping,strength),'gpu'
        from reference_color_app.lut import apply_mapping
        return apply_mapping(rgb,mapping,strength),'cpu'
