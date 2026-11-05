import unittest
import numpy as np
from dense_features import extract_dense

class DenseTests(unittest.TestCase):
 def setUp(self):
  self.rgb=np.random.default_rng(9).integers(0,256,(40,50,3),dtype=np.uint8)
  self.mask=np.zeros((40,50));self.mask[:10]=1
 def test_joint_permutation_does_not_change_packet(self):
  ix=np.random.default_rng(7).permutation(2000)
  for mode in (0,1):
   a=extract_dense(self.rgb,mode,self.mask,.5)
   b=extract_dense(self.rgb.reshape(-1,3)[ix].reshape(40,50,3),mode,self.mask.ravel()[ix].reshape(40,50),.5)
   np.testing.assert_allclose(a.values,b.values,atol=1e-6,rtol=1e-6)
 def test_repeated_pixels_preserve_distribution(self):
  for mode in (0,1):
   a=extract_dense(self.rgb,mode,self.mask,.5)
   b=extract_dense(np.repeat(self.rgb,2,axis=0),mode,np.repeat(self.mask,2,axis=0),.5)
   np.testing.assert_allclose(a.values,b.values,atol=1e-6,rtol=1e-6)
 def test_missing_regions_fall_back_to_unweighted_dense(self):
  for mode in (0,1):
   a=extract_dense(self.rgb,mode,None,0)
   for mask in (None,np.zeros((40,50)),np.full((40,50),.5)):
    np.testing.assert_array_equal(a.values,extract_dense(self.rgb,mode,mask,.5).values)
 def test_packet_layout_finite_and_input_unchanged(self):
  before=self.rgb.copy()
  for mode,n in ((0,63),(1,57)):
   result=extract_dense(self.rgb,mode,self.mask,.25)
   self.assertEqual(result.values.shape,(n,));self.assertTrue(np.isfinite(result.values).all())
  np.testing.assert_array_equal(before,self.rgb)
 def test_invalid_inputs(self):
  for mode in (True,2):
   with self.assertRaises(ValueError):extract_dense(self.rgb,mode,self.mask,.5)
  with self.assertRaises(ValueError):extract_dense(self.rgb,0,np.ones((2,2)),.5)

if __name__=='__main__':unittest.main()
