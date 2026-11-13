import importlib
import importlib.util
import unittest
import torch


class PCATests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_kd'))
        return importlib.import_module('pca_kd')

    def test_zca_identity_and_finite_constant(self):
        api=self.api();torch.manual_seed(32)
        content=torch.randn(1,4,8,8)
        torch.testing.assert_close(api.zca(content,content),content,atol=2e-5,rtol=2e-5)
        result=api.zca(torch.ones_like(content),torch.full_like(content,.4))
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result,torch.full_like(result,.4),atol=1e-5,rtol=1e-5)

    def test_high_frequency_residual_has_inclusive_edge_padding(self):
        api=self.api()
        image=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        residual=api.high_frequency(image)
        self.assertAlmostEqual(float(residual[0,0,0,0]),-15/9,places=5)
        self.assertAlmostEqual(float(residual[0,0,1,1]),0,places=5)

    def test_invalid_zca_shape_or_nonfinite_is_rejected(self):
        api=self.api()
        for a,b in ((torch.ones(1,3,4,4),torch.ones(1,4,4,4)),
                     (torch.full((1,3,4,4),float('nan')),torch.ones(1,3,4,4))):
            with self.assertRaises(ValueError):api.zca(a,b)


if __name__=='__main__':unittest.main()
