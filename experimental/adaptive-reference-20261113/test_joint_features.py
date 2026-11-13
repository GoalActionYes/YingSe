import importlib
import importlib.util
from types import SimpleNamespace
import unittest
import torch


class JointFeatureTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('joint_features'))
        return importlib.import_module('joint_features')

    def test_opposite_luma_chroma_dependence_is_not_lost(self):
        api=self.api()
        a=torch.tensor([[[[.25,.25],[.75,.75]],[[.12,.12],[-.12,-.12]],[[-.1,-.1],[.1,.1]]]])
        b=a.clone();b[:,1:]=-b[:,1:]
        torch.testing.assert_close(a.flatten(2).sort(-1).values,b.flatten(2).sort(-1).values)
        first=api.conditional_color(a,num_bins=4,prior_count=0)
        second=api.conditional_color(b,num_bins=4,prior_count=0)
        self.assertEqual(tuple(first.shape),(1,3,4))
        self.assertGreater(float((first-second).abs().max()),.4)
        torch.testing.assert_close(first[0,:2,1],torch.tensor([.24,-.2]))
        torch.testing.assert_close(first[0,:,0],torch.zeros(3))
        shrunk=api.conditional_color(a,num_bins=4,prior_count=32)
        torch.testing.assert_close(shrunk,first*(2/34))

    def test_invalid_values_are_rejected_before_binning(self):
        api=self.api();valid=torch.zeros(1,3,2,2)
        for value in [valid.fill_(float('nan')),torch.zeros(1,4,2,2),torch.zeros(1,3,0,2),torch.zeros(1,3,2,2,dtype=torch.int64)]:
            with self.assertRaises(ValueError):api.conditional_color(value)
        for bins,prior in [(True,32),(0,32),(128,-1),(128,float('nan'))]:
            with self.assertRaises(ValueError):api.conditional_color(torch.zeros(1,3,2,2),bins,prior)

    def test_zero_initialized_extra_inputs_preserve_original_projection(self):
        api=self.api();torch.manual_seed(711)
        old=torch.nn.Conv1d(6,16,5,stride=2,padding=2).double()
        model=SimpleNamespace(hist_projection=torch.nn.Sequential(old))
        original=torch.randn(2,6,128,dtype=torch.float64);extra=torch.randn_like(original)
        expected=model.hist_projection(original)
        api.enable_joint_input(model)
        torch.testing.assert_close(model.hist_projection(torch.cat([original,extra],1)),expected,atol=1e-12,rtol=1e-12)
        self.assertEqual(model.hist_projection[0].in_channels,12)
        self.assertEqual(int(torch.count_nonzero(model.hist_projection[0].weight[:,6:])),0)
        with self.assertRaises(ValueError):api.enable_joint_input(model)


if __name__=='__main__':unittest.main()
