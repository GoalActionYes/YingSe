import unittest,importlib
import torch


class FieldSupervisionTests(unittest.TestCase):
    def test_latent_field_matches_original_forward_and_backpropagates(self):
        a=importlib.import_module('regional_field_supervision')
        from regional_correction import RegionalCorrection
        h=RegionalCorrection();x=torch.rand(2,3,8,8);p=torch.rand(2,1,8,8);d=torch.rand(2,1154)
        with torch.no_grad():h.layers[-1].bias.normal_(0,.2)
        field=a.latent_field(h,x,p,d,d)
        self.assertTrue(torch.allclose(h(x,x,p,d,d),x+.1*torch.tanh(field),atol=1e-7))
        self.assertEqual(float(a.field_loss(field,field.detach())),0)
        loss=a.field_loss(field,field.detach()+1);loss.backward()
        self.assertGreater(float(h.layers[-1].weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
