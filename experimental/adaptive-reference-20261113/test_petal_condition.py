import unittest
import torch


class ConditionTests(unittest.TestCase):
    def test_zero_output_and_gradients_stop_at_frozen_features(self):
        from petal_condition import ConditionHead
        head=ConditionHead();a=torch.randn(2,1024,requires_grad=True);b=torch.randn(2,1024,requires_grad=True)
        out=head(a,b);self.assertTrue(torch.equal(out,torch.zeros(2,128)))
        out.sum().backward();self.assertIsNone(a.grad);self.assertIsNone(b.grad)
        self.assertGreater(float(head.layers[-1].bias.grad.abs().sum()),0)

    def test_invalid_features_are_rejected_before_linear(self):
        from petal_condition import ConditionHead
        head=ConditionHead()
        for a,b in [(torch.zeros(1,1024),torch.ones(1,1024)),(torch.ones(1,128),torch.ones(1,128)),(torch.ones(1,1024),torch.ones(2,1024)),(torch.full((1,1024),float('nan')),torch.ones(1,1024))]:
            with self.assertRaises(ValueError):head(a,b)

    def test_freeze_keeps_head_trainable_and_backbone_identical(self):
        from petal_condition import freeze_and_create
        model=torch.nn.Linear(3,4);before={k:v.clone() for k,v in model.state_dict().items()}
        head=freeze_and_create(model)
        self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in head.parameters()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
