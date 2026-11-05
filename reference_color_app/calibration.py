"""Product-side neutral-response calibration, independent of recovered math.

Corrected = exact identity + raw reference response - raw self response.
This is our reconstruction repair policy, not a claim about Jianying internals.
"""
import numpy as np
from .contracts import Mapping
from .lut import _layers

def calibrate(mapping,neutral):
 raw=_layers(mapping);self_response=_layers(neutral)
 axis=np.linspace(0,1,64,dtype=np.float32)
 identity=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
 layers=np.clip(identity[None]+(raw-self_response),0,1).astype(np.float32)
 return Mapping(layers,dict(mapping.diagnostics,neutral_calibration=dict(
  version='relative-self-response-v1',policy='identity + reference response - target self response',
  recovered_native_policy=False,raw_diagnostics_preserved=True)))
