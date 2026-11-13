"""Research-only coefficient EMA with explicit reset on large scene changes."""
import numbers
import numpy as np
from PIL import Image
import torch
from spatial_render import PreparedSpatial,validate_rgb
from training_checks import finite_output


class TemporalGrid:
    def __init__(self,alpha=.2,cut_threshold=.16):
        for value in [alpha,cut_threshold]:
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not np.isfinite(value) or not 0<value<=1:raise ValueError('Temporal parameters must be finite within (0,1]')
        self.alpha=float(alpha);self.cut_threshold=float(cut_threshold);self.reset()

    def reset(self):self.previous=None;self.thumbnail=None;self.engine=None;self.dimensions=None

    @torch.inference_mode()
    def smooth(self,prepared,frame):
        validate_rgb(frame);grid=finite_output(prepared.grid,(1,13,8,16,16))
        thumbnail=np.array(Image.fromarray(frame).resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)/255
        difference=float(np.abs(thumbnail-self.thumbnail).mean()) if self.thumbnail is not None else None
        reason='first_frame' if self.previous is None else 'engine_changed' if self.engine is not prepared.engine else 'dimensions_changed' if self.dimensions!=frame.shape else 'scene_cut' if difference>self.cut_threshold else None
        if reason is None:
            if self.previous.device!=grid.device or self.previous.dtype!=grid.dtype:raise ValueError('Temporal grid device or dtype changed')
            result=self.previous*(1-self.alpha)+grid*self.alpha
        else:result=grid.clone()
        finite_output(result,grid.shape);self.previous=result.detach().clone();self.thumbnail=thumbnail;self.engine=prepared.engine;self.dimensions=frame.shape
        diagnostics=dict(prepared.diagnostics,temporal_alpha=self.alpha,temporal_reset_reason=reason,temporal_scene_difference=difference)
        return PreparedSpatial(prepared.engine,result,diagnostics)
