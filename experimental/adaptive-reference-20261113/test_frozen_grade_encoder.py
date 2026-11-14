import importlib
import importlib.util
import unittest
import torch
import torch.nn.functional as F


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,64,1);self.bn=torch.nn.BatchNorm2d(64)
    def forward(self,x):return F.normalize(self.bn(self.conv(x)).mean((-2,-1)),dim=1)


class FrozenGradeTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('frozen_grade_encoder'))
        return importlib.import_module('frozen_grade_encoder')
    def test_eval_and_batchnorm_buffers_stay_frozen(self):
        api=self.api();encoder=TinyEncoder();model=api.FrozenGradeEncoder(encoder,{});before={k:v.clone() for k,v in encoder.state_dict().items()};model.train(True);out=model(torch.rand(2,3,8,9))
        self.assertFalse(model.training);self.assertFalse(encoder.training);self.assertEqual(out.shape,(2,64));self.assertTrue(torch.isfinite(out).all())
        for k,v in encoder.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
    def test_trainable_head_can_backpropagate_from_frozen_vectors(self):
        model=self.api().FrozenGradeEncoder(TinyEncoder(),{});x=torch.rand(2,3,8,8,requires_grad=True);features=model(x);head=torch.nn.Linear(64,1);head(features).sum().backward()
        self.assertIsNone(x.grad);self.assertGreater(head.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
    def test_invalid_rgb_and_encoder_output_rejected(self):
        model=self.api().FrozenGradeEncoder(TinyEncoder(),{})
        for bad in [torch.rand(2,4,4),torch.rand(2,4,4,4),torch.rand(2,3,4,4,dtype=torch.float64),torch.full((2,3,4,4),1.1),torch.full((2,3,4,4),float('nan'))]:
            with self.assertRaises(ValueError):model(bad)
        broken=TinyEncoder();broken.forward=lambda x:torch.zeros(x.shape[0],63)
        with self.assertRaises(ValueError):self.api().FrozenGradeEncoder(broken,{})(torch.rand(2,3,4,4))


if __name__=='__main__':unittest.main()
