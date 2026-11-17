import unittest
import torch
from lab_objective import lab,color_objective


class LabObjectiveTests(unittest.TestCase):
    def test_black_white_neutral_and_finite_boundary_gradient(self):
        p=torch.tensor([0.,.003,.04,.04045,.041,.5,1.],dtype=torch.float64).reshape(1,1,1,-1).repeat(1,3,1,1).requires_grad_()
        value=lab(p);torch.testing.assert_close(value[:,1:],torch.zeros_like(value[:,1:]),atol=1e-4,rtol=0)
        self.assertAlmostEqual(float(value[0,0,0,0].detach()),0,places=8);self.assertAlmostEqual(float(value[0,0,0,-1].detach()),100,places=5)
        self.assertTrue(bool((value[:,0,:,-1]>value[:,0,:,0]).all()))
        value.square().sum().backward();self.assertTrue(bool(torch.isfinite(p.grad).all()))

    def test_finite_difference_and_identity_loss(self):
        p=(torch.rand(1,3,2,2,dtype=torch.float64)*.6+.2).requires_grad_();q=torch.full_like(p,.3)
        self.assertTrue(torch.autograd.gradcheck(lambda x:color_objective(x,q),(p,),eps=1e-6,atol=1e-5,rtol=1e-4))
        self.assertEqual(float(color_objective(p.detach(),p.detach())),0)
        with self.assertRaises(ValueError):lab(torch.full((1,3,2,2),float('nan')))
