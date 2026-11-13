import importlib
import importlib.util
import unittest
import torch


class ReferenceContextTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('reference_context'))
        return importlib.import_module('reference_context')
    def test_zero_context_preserves_baseline_but_head_can_learn(self):
        api=self.api();torch.manual_seed(163)
        adapter=api.ReferenceContext().double();a=torch.randn(2,128,4,4,dtype=torch.float64);b=torch.randn_like(a)
        result=adapter(a,b)
        torch.testing.assert_close(result,torch.zeros(2,128,dtype=torch.float64),atol=0,rtol=0)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(adapter.layers[-1].weight.grad).all()))
        self.assertGreater(int(torch.count_nonzero(adapter.layers[-1].weight.grad)),0)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()),65792)
    def test_invalid_features_rejected(self):
        api=self.api();adapter=api.ReferenceContext();valid=torch.zeros(1,128,4,4)
        for other in [torch.zeros(2,128,4,4),torch.zeros(1,127,4,4),torch.zeros(1,128,0,4),torch.full_like(valid,float('nan'))]:
            with self.assertRaises(ValueError):adapter(valid,other)
    def test_repeated_installation_is_rejected(self):
        api=self.api();base=torch.nn.Linear(2,2).double();api.enable_reference_context(base)
        self.assertEqual(base.reference_adapter.layers[0].weight.dtype,torch.float64)
        with self.assertRaises(ValueError):api.enable_reference_context(base)


if __name__=='__main__':unittest.main()
