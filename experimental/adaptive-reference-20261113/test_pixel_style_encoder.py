import unittest
import torch
from pixel_style_encoder import PixelStyleEncoder


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
        self.bn = torch.nn.BatchNorm2d(2)

    def forward(self, x):
        return {'out': self.bn(self.conv(x))}


class PixelEncoderTests(unittest.TestCase):
    def test_style_backbone_learns_without_mutating_frozen_mask_backbone(self):
        torch.manual_seed(51)
        source = TinyBackbone().eval()
        for p in source.parameters():
            p.requires_grad_(False)
        old = {k: v.clone() for k, v in source.state_dict().items()}
        model = PixelStyleEncoder(source, feature_channels=2)
        model.train()
        self.assertFalse(model.backbone.bn.training)
        rgb = torch.rand(3, 3, 8, 8)
        stats = torch.rand(3, 42)
        mask = torch.rand(3, 1, 8, 8)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        prediction = model(rgb, stats, mask)
        self.assertEqual(tuple(prediction.shape), (3, 2, 6))
        (prediction-.4).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(model.backbone.conv.weight.grad).all()))
        self.assertTrue(bool((model.backbone.conv.weight.grad != 0).any()))
        optimizer.step()
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, old[name], atol=0, rtol=0)
        torch.testing.assert_close(model.backbone.bn.running_mean, old['bn.running_mean'], atol=0, rtol=0)

    def test_constant_style_features_have_finite_moment_backward(self):
        source = TinyBackbone().eval()
        with torch.no_grad():
            source.conv.weight.zero_()
            source.conv.bias.zero_()
        model = PixelStyleEncoder(source, feature_channels=2).train()
        output = model(torch.zeros(2, 3, 8, 8), torch.zeros(2, 42), torch.zeros(2, 1, 8, 8))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
