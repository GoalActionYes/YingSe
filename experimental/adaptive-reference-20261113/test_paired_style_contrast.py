import importlib
import importlib.util
import unittest
import torch


class PairedStyleContrastTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('paired_style_contrast'))
        return importlib.import_module('paired_style_contrast')
    def test_correct_order_better_than_within_pair_swap(self):
        api=self.api();ref=torch.eye(4,dtype=torch.float64);gt=ref.clone()
        correct=api.paired_style_contrast(ref,gt);wrong=api.paired_style_contrast(ref,gt[[1,0,3,2]])
        self.assertLess(correct.item(),.001);self.assertGreater(wrong.item(),9.)
    def test_no_cross_content_negatives(self):
        api=self.api();ref=torch.eye(2,dtype=torch.float64).repeat(2,1)
        self.assertLess(api.paired_style_contrast(ref,ref).item(),.001)
        uniform=torch.ones((4,2),dtype=torch.float64)
        self.assertAlmostEqual(api.paired_style_contrast(uniform,uniform).item(),float(torch.log(torch.tensor(2.,dtype=torch.float64))),places=10)
    def test_both_branches_receive_finite_gradients(self):
        api=self.api();torch.manual_seed(7);a=torch.randn(4,5,dtype=torch.float64,requires_grad=True);b=torch.randn(4,5,dtype=torch.float64,requires_grad=True);loss=api.paired_style_contrast(a,b);loss.backward()
        for v in [a,b]:self.assertTrue(torch.isfinite(v.grad).all());self.assertGreater(v.grad.abs().sum().item(),0)
    def test_invalid_inputs_rejected(self):
        api=self.api();a=torch.eye(4)
        for b in [torch.eye(3),torch.full((4,4),float('nan')),torch.ones((4,4),dtype=torch.int64)]:
            with self.assertRaises(ValueError):api.paired_style_contrast(a,b)
        with self.assertRaises(ValueError):api.paired_style_contrast(a,a,temperature=0)


if __name__=='__main__':unittest.main()
