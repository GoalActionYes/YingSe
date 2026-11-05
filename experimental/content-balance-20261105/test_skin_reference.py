import unittest
import numpy as np
from skin_reference import match_skin

class ReferenceSkinTests(unittest.TestCase):
 def setUp(self):
  self.a=np.full((20,20,3),[140,110,90],np.uint8)
  self.b=np.full((20,20,3),[130,120,100],np.uint8)
  self.m=np.zeros((20,20));self.m[:10]=1
 def test_moves_chroma_towards_reference(self):
  out,d=match_skin(self.a,self.b,self.m,self.m)
  self.assertTrue(d['applied'])
  def c(x):return x-x@np.array([.299,.587,.114])
  self.assertLess(np.linalg.norm(c(out[0,0].astype(float))-c(self.b[0,0].astype(float))),
                  np.linalg.norm(c(self.a[0,0].astype(float))-c(self.b[0,0].astype(float))))
 def test_background_is_exact_and_delta_bounded(self):
  out,_=match_skin(self.a,self.b,self.m,self.m)
  np.testing.assert_array_equal(out[10:],self.a[10:])
  self.assertLessEqual(np.abs(out.astype(float)-self.a).max(),8.5)
 def test_luminance_preserved_even_near_gamut_boundary(self):
  a=self.a.copy();a[0,0]=[255,10,0]
  out,_=match_skin(a,self.b,self.m,self.m)
  self.assertLessEqual(np.abs((out.astype(float)-a)@np.array([.299,.587,.114])).max(),.501)
 def test_identity_and_missing_region(self):
  for ref,mask in ((self.a,self.m),(self.b,np.zeros_like(self.m))):
   out,_=match_skin(self.a,ref,self.m,mask)
   np.testing.assert_array_equal(out,self.a)
 def test_invalid_mask_rejected(self):
  with self.assertRaises(ValueError):match_skin(self.a,self.b,self.m,np.full((20,20),np.nan))

if __name__=='__main__':unittest.main()
