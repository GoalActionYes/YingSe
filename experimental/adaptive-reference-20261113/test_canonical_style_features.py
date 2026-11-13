import importlib
import importlib.util
import unittest
import torch
from torch import nn
import torch.nn.functional as F


class CanonicalFeatureTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('canonical_style_features'))
        return importlib.import_module('canonical_style_features')

    def test_global_pool_matches_original_style_projection(self):
        api = self.api()
        torch.manual_seed(5)
        feature = torch.randn(2,8,7,7)
        person = torch.rand(2,1,224,224)
        head = nn.Sequential(nn.Linear(8,4),nn.GELU(),nn.Linear(4,4))
        result = api.project_regions(feature,person,head).reshape(2,3,4)
        expected = F.normalize(head(feature.mean(dim=(2,3))),dim=-1)
        torch.testing.assert_close(result[:,0],expected,atol=1e-6,rtol=1e-6)

    def test_missing_person_has_finite_projection_and_background_matches_global(self):
        api = self.api()
        features = torch.ones(2,8,7,7)
        result = api.project_regions(features,torch.zeros(2,1,224,224),nn.Linear(8,4)).reshape(2,3,4)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[:,0],result[:,2])

    def test_invalid_probability_is_rejected(self):
        api = self.api()
        with self.assertRaises(ValueError):
            api.project_regions(torch.ones(2,8,7,7),torch.full((2,1,224,224),float('nan')),nn.Linear(8,4))


if __name__ == '__main__': unittest.main()
