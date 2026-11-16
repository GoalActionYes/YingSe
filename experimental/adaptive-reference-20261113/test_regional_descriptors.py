import unittest
import importlib
import numpy as np


class RegionalDescriptorTests(unittest.TestCase):
    def test_single_pixel_weighting_and_full_mass(self):
        api=importlib.import_module('regional_descriptors')
        rgb=np.array([[[30,50,70],[180,150,130]]],np.uint8);p=np.array([[1.,0.]],np.float32)
        a=api.describe(rgb,p);self.assertEqual(a.shape,(1154,))
        h=api.histogram(rgb,np.ones(p.shape));np.testing.assert_allclose(h.sum(1),1)
        np.testing.assert_array_equal(a[:384].reshape(3,128),h)
        np.testing.assert_array_equal(a[384:768].reshape(3,128),api.histogram(rgb[:,:1],np.ones((1,1))))
        np.testing.assert_allclose(a[-2:],[.5,.5])

    def test_empty_region_has_zero_descriptor_and_full_background_matches_global(self):
        api=importlib.import_module('regional_descriptors');rgb=np.full((8,8,3),128,np.uint8);p=np.zeros((8,8),np.float32)
        before=rgb.copy();a=api.describe(rgb,p)
        self.assertTrue((a[384:768]==0).all());np.testing.assert_array_equal(a[768:1152],a[:384]);np.testing.assert_array_equal(rgb,before)

    def test_invalid_probability_dtype_and_nonfinite_rejected(self):
        api=importlib.import_module('regional_descriptors');rgb=np.full((8,8,3),128,np.uint8)
        for p in [np.zeros((4,8)),np.full((8,8),np.nan),np.full((8,8),-1),np.full((8,8),2)]:
            with self.assertRaises(ValueError):api.describe(rgb,p)
        with self.assertRaises(ValueError):api.describe(rgb.astype(float),np.zeros((8,8)))


if __name__=='__main__':unittest.main()
