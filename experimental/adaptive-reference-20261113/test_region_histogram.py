import unittest
import numpy as np
import torch


class RegionHistogramTests(unittest.TestCase):
    def test_uniform_matches_manual_author_bins(self):
        from region_histogram import weighted_histogram
        x=torch.rand(1,3,11,13,generator=torch.Generator().manual_seed(42))
        x[:,1:]-=.5
        shifted=x.clone();shifted[:,1:]+=.5
        expected=torch.stack([torch.bincount((c.flatten()*128).floor().clamp(0,127).long(),minlength=128) for c in shifted[0]])[None]/143
        torch.testing.assert_close(weighted_histogram(x,torch.ones(1,1,11,13)),expected)

    def test_one_pixel_and_joint_permutation(self):
        from region_histogram import weighted_histogram
        x=torch.tensor([[[[.1,.9]],[[0.,.4]],[[-.4,0.]]]])
        w=torch.tensor([[[[0.,1.]]]])
        h=weighted_histogram(x,w)
        self.assertEqual(h[0,0,115].item(),1.)
        self.assertEqual(h[0,1,115].item(),1.)
        self.assertEqual(h[0,2,64].item(),1.)
        torch.testing.assert_close(h,weighted_histogram(x.flip(-1),w.flip(-1)))

    def test_support_rejects_no_person_and_uncertain_dust(self):
        from region_histogram import supported
        p=np.zeros((10,10),np.float32)
        self.assertFalse(supported(p,p))
        self.assertFalse(supported(p+.1,p+.1))
        p[:5]=1
        self.assertTrue(supported(p,p))
        self.assertFalse(supported(p,np.ones_like(p)))

    def test_blend_endpoints_and_no_mutation(self):
        from region_histogram import blend
        a=np.full((3,4,3),.2,np.float32);b=np.full_like(a,.8);saved=a.copy()
        np.testing.assert_array_equal(blend(a,b,np.zeros((3,4),np.float32)),np.full((3,4,3),204,np.uint8))
        np.testing.assert_array_equal(blend(a,b,np.ones((3,4),np.float32)),np.full((3,4,3),51,np.uint8))
        np.testing.assert_array_equal(a,saved)

    def test_invalid_mask_and_zero_mass_fail_closed(self):
        from region_histogram import weighted_histogram,supported,blend
        x=torch.zeros(1,3,3,4)
        for mask in [torch.zeros(1,1,3,4),torch.full((1,1,3,4),float('nan')),torch.ones(1,1,3,4)*2]:
            with self.assertRaises(ValueError):weighted_histogram(x,mask)
        with self.assertRaises(ValueError):supported(np.full((3,4),np.nan),np.zeros((3,4)))
        with self.assertRaises(ValueError):blend(np.zeros((3,4,3)),np.ones((3,4,3)),np.ones((4,4)))
