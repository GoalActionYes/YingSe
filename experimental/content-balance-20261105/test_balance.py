import unittest
import numpy as np
from balance import sample_atlas,balanced_share

class BalanceTests(unittest.TestCase):
 def test_zero_keeps_measured_share(self):
  for p in (.01,.1,.5,.9,.99):self.assertEqual(balanced_share(p,0),p)
 def test_extreme_regions_cannot_be_arbitrarily_amplified(self):
  for p in (.01,.1,.4,.9,.99):
   q=balanced_share(p,.5)
   self.assertLessEqual(q/p,2+1e-12);self.assertLessEqual((1-q)/(1-p),2+1e-12)
   self.assertGreaterEqual(q/p,.5-1e-12);self.assertGreaterEqual((1-q)/(1-p),.5-1e-12)
 def test_sampling_zero_is_deterministic_and_input_untouched(self):
  rgb=np.arange(600,dtype=np.uint8).reshape(10,20,3);before=rgb.copy();m=np.zeros((10,20),np.float32);m[:4]=1
  a,d=sample_atlas(rgb,m,0,side=40);b,e=sample_atlas(rgb,m,0,side=40)
  np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(rgb,before)
  self.assertEqual(d['source_share'],d['balanced_share'])
 def test_uniform_person_or_background_falls_back(self):
  rgb=np.full((20,20,3),120,np.uint8)
  for m in (None,np.zeros((20,20)),np.ones((20,20))):
   a,d=sample_atlas(rgb,m,.5);self.assertFalse(d['applied']);self.assertEqual(a.shape,(320,320,3))
 def test_only_quota_changes_not_palette(self):
  rgb=np.zeros((20,20,3),np.uint8);rgb[:2]=[200,100,90];rgb[2:]=[20,90,180];m=np.zeros((20,20));m[:2]=1
  a,d=sample_atlas(rgb,m,.5,side=100)
  self.assertAlmostEqual(float((a[...,0]==200).mean()),d['balanced_share'],places=4)
  self.assertEqual(len(np.unique(a.reshape(-1,3),axis=0)),2)
 def test_invalid_parameters_rejected(self):
  for value in (-1,1,True,float('nan')):
   with self.assertRaises(ValueError):balanced_share(.3,value)
  with self.assertRaises(ValueError):sample_atlas(np.ones((3,3,3),np.uint8),np.zeros((2,2)),.25)
 def test_mostly_uncertain_mask_cannot_drive_statistics(self):
  rgb=np.full((50,50,3),120,np.uint8);m=np.full((50,50),.5);m[:2]=1;m[-2:]=0
  _,d=sample_atlas(rgb,m,.5);self.assertFalse(d['applied'])

if __name__=='__main__':unittest.main()
