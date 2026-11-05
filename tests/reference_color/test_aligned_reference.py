import unittest
import numpy as np
from reference_color_app.aligned_reference import fit_aligned,make_mapping
from reference_color_app.lut import apply_mapping

class AlignedTests(unittest.TestCase):
 def setUp(self):self.a=np.random.default_rng(71).integers(10,245,(80,96,3),dtype=np.uint8)
 def test_learns_global_warm_transform(self):
  b=np.rint(np.clip(self.a*[1.04,.98,.92]+[3,1,0],0,255)).astype(np.uint8)
  fit=fit_aligned(self.a,b);self.assertTrue(fit['accepted'])
  out=apply_mapping(self.a,make_mapping(fit),1)
  self.assertLess(np.abs(out.astype(float)-b).mean(),.6)
 def test_same_histogram_different_layout_is_rejected(self):
  b=self.a.reshape(-1,3)[::-1].reshape(self.a.shape)
  self.assertFalse(fit_aligned(self.a,b)['accepted'])
 def test_flat_image_is_not_alignment_evidence(self):
  a=np.full((80,96,3),120,np.uint8)
  self.assertFalse(fit_aligned(a,a)['accepted'])
 def test_shift_is_rejected(self):
  self.assertFalse(fit_aligned(self.a,np.roll(self.a,5,axis=1))['accepted'])
 def test_small_local_content_change_is_rejected(self):
  b=self.a.copy();b[8:16,8:16]=255-b[8:16,8:16]
  self.assertFalse(fit_aligned(self.a,b)['accepted'])
 def test_input_unchanged_and_strength_zero(self):
  before=self.a.copy();fit=fit_aligned(self.a,self.a)
  np.testing.assert_array_equal(apply_mapping(self.a,make_mapping(fit),0),self.a)
  np.testing.assert_array_equal(before,self.a)

if __name__=='__main__':unittest.main()
