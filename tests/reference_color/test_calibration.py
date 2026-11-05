import unittest
import numpy as np
from reference_color_app.contracts import Mapping
from reference_color_app.lut import apply_mapping
from reference_color_app.calibration import calibrate

class CalibrationTests(unittest.TestCase):
 def test_same_response_is_identity_at_all_strengths(self):
  rng=np.random.default_rng(31)
  layers=rng.random((2,64,64,64,3),dtype=np.float32)
  mapping=Mapping(layers,{})
  fixed=calibrate(mapping,mapping)
  pixels=rng.integers(0,256,(100,100,3),dtype=np.uint8)
  for strength in (0,.25,.5,.8,.80002,1):
   np.testing.assert_array_equal(apply_mapping(pixels,fixed,strength),pixels)
 def test_retains_relative_reference_change(self):
  axis=np.linspace(0,1,64,dtype=np.float32);identity=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
  neutral=Mapping(np.stack([identity*.8]*2),{})
  desired=Mapping(np.clip(neutral.layers+.05,0,1),{})
  out=calibrate(desired,neutral)
  np.testing.assert_allclose(out.layers,np.clip(np.stack([identity]*2)+.05,0,1),atol=1e-7)
 def test_does_not_mutate_inputs(self):
  layers=np.zeros((2,64,64,64,3),np.float32);m=Mapping(layers,{'raw':True})
  calibrate(m,m);self.assertEqual(layers.max(),0);self.assertEqual(m.diagnostics,{'raw':True})

if __name__=='__main__':unittest.main()
