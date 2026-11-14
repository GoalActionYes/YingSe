import importlib
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
import torch


class RadiusContextTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('radius_context'))
        return importlib.import_module('radius_context')
    def test_fixed_norm_direction_and_positive_scale_invariance(self):
        api=self.api();torch.manual_seed(2);x=torch.randn(3,128,4,5,dtype=torch.float64);y=api.normalized_map(x,7.)
        self.assertEqual(y.shape,(3,128,1,1));self.assertTrue(torch.allclose(y.flatten(1).norm(dim=1),torch.full((3,),7.,dtype=torch.float64)))
        self.assertTrue(torch.allclose(y,api.normalized_map(x*3,7.),atol=1e-12,rtol=0))
        cosine=torch.nn.functional.cosine_similarity(y.flatten(1),x.mean((-2,-1)),dim=1);self.assertTrue(torch.allclose(cosine,torch.ones_like(cosine)))
    def test_zero_and_directional_gradients(self):
        api=self.api();z=api.normalized_map(torch.zeros(2,128,3,3),5.);self.assertTrue(torch.equal(z,torch.zeros_like(z)))
        torch.manual_seed(3);x=torch.randn(2,128,3,3,dtype=torch.float64,requires_grad=True);weights=torch.arange(128,dtype=torch.float64)[None,:,None,None];(api.normalized_map(x,5.)*weights).sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
    def test_bad_maps_and_radius_rejected(self):
        api=self.api();x=torch.ones(2,128,3,3)
        for radius in [0,-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):api.normalized_map(x,radius)
        for bad in [torch.ones(2,127,3,3),torch.ones(2,128,3),torch.full_like(x,float('nan')),x.half()]:
            with self.assertRaises(ValueError):api.normalized_map(bad,2.)
    def test_fixed_train_median(self):
        api=self.api();self.assertEqual(api.calibrated_radius([1,2,3,4]),2.5)
        for invalid in [[],[0,0],[float('nan')],[-1,2],[float('inf')]]:
            with self.assertRaises(ValueError):api.calibrated_radius(invalid)
    def test_raw_mode_delegates_original_function(self):
        api=self.api();sentinel=object()
        with patch.object(api,'appearance_forward',return_value=sentinel) as f:
            self.assertIs(api.radius_forward('m','api','c','r',7.,normalized=False),sentinel);f.assert_called_once_with('m','api','c','r')


if __name__=='__main__':unittest.main()
