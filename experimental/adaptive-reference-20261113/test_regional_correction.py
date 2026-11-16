import unittest,importlib
import torch


class CorrectionTests(unittest.TestCase):
    def test_zero_and_disabled_restore_float_baseline(self):
        a=importlib.import_module('regional_correction');head=a.RegionalCorrection()
        x=torch.rand(2,3,8,8);base=x*.7;p=torch.rand(2,1,8,8);d=torch.rand(2,1154)
        self.assertTrue(torch.equal(head(x,base,p,d,d),base))
        with torch.no_grad():head.layers[-1].bias.fill_(1)
        self.assertTrue(torch.equal(head(x,base,p,d,d,enabled=False),base))

    def test_affine_probability_endpoints_and_bound(self):
        a=importlib.import_module('regional_correction');x=torch.zeros(1,3,4,4);base=x.clone();coef=torch.zeros(1,3,3,4)
        coef[:,1,:,3]=1;coef[:,2,:,3]=-1
        hi=a.apply_residual(x,base,torch.ones(1,1,4,4),coef)
        lo=a.apply_residual(x,base,torch.zeros(1,1,4,4),coef)
        self.assertTrue(torch.all(hi>0));self.assertTrue(torch.all(lo<0));self.assertTrue(torch.all(hi.abs()<=.1))
        self.assertTrue(torch.equal(a.apply_residual(x,base,torch.full((1,1,4,4),.5),coef),base))

    def test_global_branch_operates_without_people_and_gradients_reach_head(self):
        a=importlib.import_module('regional_correction');head=a.RegionalCorrection();x=torch.rand(1,3,8,8);p=torch.zeros(1,1,8,8);d=torch.rand(1,1154)
        result=head(x,x,p,d,d);result.sum().backward()
        self.assertGreater(float(head.layers[-1].weight.grad.abs().sum()),0)
        with torch.no_grad():head.layers[-1].bias[3]=1
        self.assertFalse(torch.equal(head(x,x,p,d,d),x))

    def test_control_uses_full_histograms_and_preserves_support(self):
        a=importlib.import_module('regional_correction');d=torch.rand(2,1154);before=d.clone();c=a.global_control(d)
        self.assertTrue(torch.equal(c[:,:384],c[:,384:768]));self.assertTrue(torch.equal(c[:,:384],c[:,768:1152]))
        self.assertTrue(torch.equal(c[:,-2:],d[:,-2:]));self.assertTrue(torch.equal(before,d))

    def test_invalid_probabilities_and_nonfinite_input_rejected(self):
        a=importlib.import_module('regional_correction');head=a.RegionalCorrection();x=torch.rand(1,3,4,4);d=torch.rand(1,1154)
        for p in [torch.full((1,1,4,4),float('nan')),torch.full((1,1,4,4),2.),torch.zeros(1,1,2,4)]:
            with self.assertRaises(ValueError):head(x,x,p,d,d)


if __name__=='__main__':unittest.main()
