import unittest
import torch
from reference_normalization import ReferenceNormalization,histogram_loss


class ReferenceNormalizationTests(unittest.TestCase):
    def test_distribution_and_shared_encoder_gradient(self):
        torch.manual_seed(1);encoder=torch.nn.Conv2d(3,128,1);head=ReferenceNormalization();rgb=torch.rand(2,3,8,8)
        hist=torch.softmax(torch.randn(2,3,128),-1);out=head(encoder(rgb),hist)
        torch.testing.assert_close(out.sum(-1),torch.ones(2,3))
        histogram_loss(out,hist).backward()
        self.assertGreater(float(encoder.weight.grad.abs().sum()),0)
        self.assertGreater(float(head.layers[-1].weight.grad.abs().sum()),0)

    def test_known_formula_and_frozen_teacher(self):
        a=torch.zeros(1,3,128);b=a.clone();a[:,:,0]=1;b[:,:,1]=1
        self.assertEqual(float(histogram_loss(a,a)),0)
        expected=((a.add(1e-8).sqrt()-b.add(1e-8).sqrt()).square().sum(-1)).mean()
        torch.testing.assert_close(histogram_loss(a,b),expected)
        with self.assertRaises(ValueError):histogram_loss(a,b.requires_grad_())
        with self.assertRaises(ValueError):ReferenceNormalization()(torch.rand(1,128,2,2),torch.ones(1,3,128))
