import unittest,importlib
import torch


class BaselineContextTests(unittest.TestCase):
    def test_zero_output_and_context_source(self):
        a=importlib.import_module('semantic_baseline_context');h=a.BaselineContextCorrection();rgb=torch.full((1,3,16,16),.2);base=torch.full_like(rgb,.7);t=torch.rand(1,64,387)
        self.assertTrue(torch.equal(h(rgb,base,t,t),base))
        torch.testing.assert_close(a.color_context(rgb,base,False),torch.full((1,64,3),.7))
        torch.testing.assert_close(a.color_context(rgb,base,True),torch.full((1,64,3),.2))

    def test_nonzero_head_observes_baseline_and_has_gradient(self):
        a=importlib.import_module('semantic_baseline_context');h=a.BaselineContextCorrection();rgb=torch.rand(1,3,8,8);t=torch.rand(1,64,387)
        with torch.no_grad():
            h.layers[0].weight.zero_();h.layers[0].bias.zero_();h.layers[0].weight[0,-3:]=1
            h.layers[-1].weight.zero_();h.layers[-1].weight[:,0]=1
        b1=torch.full_like(rgb,.2);b2=torch.full_like(rgb,.6)
        r1=h(rgb,b1,t,t)-b1;r2=h(rgb,b2,t,t)-b2
        self.assertFalse(torch.allclose(r1,r2))
        c1=h(rgb,b1,t,t,control=True)-b1;c2=h(rgb,b2,t,t,control=True)-b2
        torch.testing.assert_close(c1,c2,atol=1e-7,rtol=1e-6)
        h(rgb,b1,t,t).sum().backward();self.assertGreater(float(h.layers[0].weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):h(rgb,b1*float('nan'),t,t)


if __name__=='__main__':unittest.main()
