import unittest
import numpy as np
from composition import extract_pair
from dense_features import extract_dense


class CompositionTests(unittest.TestCase):
 def setUp(self):
  self.a=np.random.default_rng(17).integers(0,256,(40,50,3),dtype=np.uint8)
  self.b=np.random.default_rng(18).integers(0,256,(40,50,3),dtype=np.uint8)
  self.am=np.zeros((40,50));self.am[:10]=1
  self.bm=np.zeros((40,50));self.bm[:30]=1
 def test_target_packet_never_changes(self):
  a,b,diag=extract_pair(self.a,self.b,self.am,self.bm,.5)
  np.testing.assert_array_equal(a.values,extract_dense(self.a,0).values)
  self.assertTrue(diag['applied'])
  self.assertAlmostEqual(b.diagnostics['balanced_share'],.5)
 def test_matching_composition_is_exact_noop(self):
  a,b,diag=extract_pair(self.a,self.b,self.am,self.am,.5)
  np.testing.assert_array_equal(a.values,extract_dense(self.a,0).values)
  np.testing.assert_array_equal(b.values,extract_dense(self.b,1).values)
  self.assertFalse(diag['applied'])
 def test_zero_and_unreliable_masks_fall_back_exactly(self):
  for am,bm,amount in ((self.am,self.bm,0),(None,self.bm,.5),(self.am,None,.5),
                       (self.am,np.ones((40,50)),.5),(np.full((40,50),.5),self.bm,.5)):
   a,b,diag=extract_pair(self.a,self.b,am,bm,amount)
   np.testing.assert_array_equal(a.values,extract_dense(self.a,0).values)
   np.testing.assert_array_equal(b.values,extract_dense(self.b,1).values)
   self.assertFalse(diag['applied'])
 def test_relative_weights_remain_bounded(self):
  am=np.zeros((40,50));am[:1]=1
  bm=1-am
  _,b,_=extract_pair(self.a,self.b,am,bm,.5)
  p=b.diagnostics['source_share'];q=b.diagnostics['balanced_share']
  self.assertGreaterEqual(q/p,.5);self.assertLessEqual(q/p,2)
  self.assertGreaterEqual((1-q)/(1-p),.5);self.assertLessEqual((1-q)/(1-p),2)
 def test_inputs_unchanged_and_invalid_mask_rejected(self):
  aa=self.a.copy();bb=self.bm.copy()
  extract_pair(self.a,self.b,self.am,self.bm,.5)
  np.testing.assert_array_equal(self.a,aa);np.testing.assert_array_equal(self.bm,bb)
  for invalid in (np.zeros((2,2)),np.full((40,50),np.nan),np.full((40,50),1.01)):
   with self.assertRaises(ValueError):extract_pair(self.a,self.b,invalid,self.bm,.5)

if __name__=='__main__':unittest.main()
