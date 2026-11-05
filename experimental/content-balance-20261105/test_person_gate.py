import unittest
import numpy as np
from person_gate import face_evidence,segment


class FaceGateTests(unittest.TestCase):
 def test_clothes_and_body_do_not_establish_face_evidence(self):
  category=np.full((64,64),4,np.uint8);confidence=np.ones((64,64),np.float32)
  self.assertFalse(face_evidence(category,confidence)['eligible'])
 def test_connected_confident_face_passes(self):
  category=np.zeros((64,64),np.uint8);category[10:14,10:14]=3
  evidence=face_evidence(category,np.ones((64,64)))
  self.assertTrue(evidence['eligible']);self.assertEqual(evidence['largest_face_component'],16)
 def test_scattered_pixels_are_not_a_face(self):
  category=np.zeros((64,64),np.uint8);category[::4,::4]=3
  evidence=face_evidence(category,np.ones((64,64)))
  self.assertFalse(evidence['eligible']);self.assertEqual(evidence['largest_face_component'],1)
 def test_low_confidence_face_rejected_without_mutation(self):
  category=np.full((64,64),3,np.uint8);confidence=np.full((64,64),.84)
  cc=category.copy();ff=confidence.copy()
  self.assertFalse(face_evidence(category,confidence)['eligible'])
  np.testing.assert_array_equal(cc,category);np.testing.assert_array_equal(ff,confidence)
 def test_minimum_area_scales_with_analysis_size(self):
  category=np.zeros((512,512),np.uint8);category[:4,:4]=3
  evidence=face_evidence(category,np.ones((512,512)))
  self.assertFalse(evidence['eligible']);self.assertEqual(evidence['minimum_face_component'],27)
 def test_invalid_model_arrays_rejected(self):
  for category,confidence in ((np.zeros((3,3)),np.ones((2,2))),
      (np.zeros((3,3)),np.full((3,3),np.nan)),(np.zeros((3,3)),np.full((3,3),1.1)),
      (np.zeros((0,0)),np.zeros((0,0)))):
   with self.assertRaises(ValueError):face_evidence(category,confidence)
 def test_invalid_rgb_rejected_before_model_loading(self):
  for rgb in (np.zeros((10,10)),np.zeros((10,10,3)),np.zeros((0,10,3),np.uint8)):
   with self.assertRaises(ValueError):segment(rgb)

if __name__=='__main__':unittest.main()
