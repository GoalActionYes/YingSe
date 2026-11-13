import unittest
import torch
from style_features import pooled_moments, color_statistics


class StyleFeatureTests(unittest.TestCase):
    def test_soft_region_weighted_moments(self):
        features = torch.tensor([[[[1., 1., 0., 0.]]]])
        masks = torch.tensor([[[[1., 1., 1., 1.]], [[1., 1., 0., 0.]], [[0., 0., 1., 1.]]]])
        mean, std = pooled_moments(features, masks)
        torch.testing.assert_close(mean[0, :, 0], torch.tensor([.5, 1., 0.]))
        torch.testing.assert_close(std[0, :, 0], torch.tensor([.5, 0., 0.]))

    def test_color_features_remain_finite_with_missing_person(self):
        rgb = torch.full((2, 3, 3, 4), .25)
        stats = color_statistics(rgb, torch.zeros(2, 1, 3, 4))
        self.assertEqual(tuple(stats.shape), (2, 42))
        self.assertTrue(bool(torch.isfinite(stats).all()))
        self.assertFalse(bool((stats[0]-stats[1]).abs().any()))


if __name__ == '__main__':
    unittest.main()
