import importlib
import importlib.util
import unittest
import torch
import torch.nn.functional as F


class GlobalChoiceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('global_choice'))
        return importlib.import_module('global_choice')

    def test_recovers_known_convex_color_combination(self):
        api=self.api(); torch.manual_seed(64)
        candidates=[torch.rand(1,3,28,28) for _ in range(3)]
        expected=sum(x*w for x,w in zip(candidates,(.2,.3,.5)))
        anchors=F.adaptive_avg_pool2d(expected,(7,7))
        output,info=api.choose_global(candidates,anchors,torch.ones(1,1,7,7))
        torch.testing.assert_close(output,expected,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(torch.tensor(info['weights']),torch.tensor([.2,.3,.5]),atol=1e-4,rtol=1e-4)

    def test_can_keep_original_and_falls_back_when_unreliable(self):
        api=self.api(); original=torch.full((1,3,28,28),.4)
        candidates=[original,original+.3,original-.2]
        # All outputs have same gray shape, but original exactly matches the anchor.
        output,info=api.choose_global(candidates,torch.full((1,3,2,2),.4),torch.ones(1,1,2,2))
        torch.testing.assert_close(output,original)
        output,info=api.choose_global(candidates,torch.full((1,3,2,2),.4),torch.zeros(1,1,2,2))
        self.assertTrue(torch.equal(output,candidates[1]))
        self.assertTrue(info['fallback'])


if __name__=='__main__': unittest.main()
