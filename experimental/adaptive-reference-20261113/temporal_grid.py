"""Research-only coefficient EMA with explicit reset on large scene changes."""
import numbers
import numpy as np
from PIL import Image
import torch
from spatial_render import PreparedSpatial,validate_rgb
from training_checks import finite_output


class TemporalGrid:
    def __init__(self,alpha=.2,cut_threshold=.16,relative_cut=False,relative_threshold=.85,minimum_change=.03,contrast_floor=.03):
        for value in [alpha,cut_threshold,minimum_change,contrast_floor]:
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not np.isfinite(value) or not 0<value<=1:raise ValueError('Temporal parameters must be finite within (0,1]')
        if type(relative_cut)!=bool or isinstance(relative_threshold,(bool,np.bool_)) or not isinstance(relative_threshold,numbers.Real) or not np.isfinite(relative_threshold) or not 0<relative_threshold<=4:raise ValueError('Invalid relative-cut policy')
        self.alpha=float(alpha);self.cut_threshold=float(cut_threshold);self.relative_cut=relative_cut;self.relative_threshold=float(relative_threshold);self.minimum_change=float(minimum_change);self.contrast_floor=float(contrast_floor);self.reset()

    def settings(self):
        result=dict(alpha=self.alpha,cut_threshold=self.cut_threshold)
        if self.relative_cut:result.update(relative_cut=True,relative_threshold=self.relative_threshold,minimum_change=self.minimum_change,contrast_floor=self.contrast_floor)
        return result

    def reset(self):self.previous=None;self.thumbnail=None;self.engine=None;self.dimensions=None

    @torch.inference_mode()
    def smooth(self,prepared,frame):
        validate_rgb(frame);grid=finite_output(prepared.grid,(1,13,8,16,16))
        thumbnail=np.array(Image.fromarray(frame).resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32)/255
        difference=float(np.abs(thumbnail-self.thumbnail).mean()) if self.thumbnail is not None else None
        scale=max(float((thumbnail.std()+self.thumbnail.std())*.5),self.contrast_floor) if self.relative_cut and self.thumbnail is not None else None
        relative=difference/scale if scale is not None else None
        relative_event=relative is not None and difference>self.minimum_change and relative>self.relative_threshold
        reason='first_frame' if self.previous is None else 'engine_changed' if self.engine is not prepared.engine else 'dimensions_changed' if self.dimensions!=frame.shape else 'scene_cut' if difference>self.cut_threshold else 'scene_cut_relative' if relative_event else None
        if reason is None:
            if self.previous.device!=grid.device or self.previous.dtype!=grid.dtype:raise ValueError('Temporal grid device or dtype changed')
            result=self.previous*(1-self.alpha)+grid*self.alpha
        else:result=grid.clone()
        finite_output(result,grid.shape);self.previous=result.detach().clone();self.thumbnail=thumbnail;self.engine=prepared.engine;self.dimensions=frame.shape
        diagnostics=dict(prepared.diagnostics,temporal_alpha=self.alpha,temporal_reset_reason=reason,temporal_scene_difference=difference,temporal_relative_difference=relative,temporal_contrast_scale=scale)
        return PreparedSpatial(prepared.engine,result,diagnostics)
