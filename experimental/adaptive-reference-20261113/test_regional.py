import unittest
import numpy as np
from regional import fit, apply


class RegionalChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.rgb = rng.integers(50, 200, (32, 48, 3), dtype=np.uint8)
        self.masks = np.zeros((32, 48, 2), np.float32)
        self.masks[:, :24, 0] = 1
        self.masks[:, 24:, 1] = 1

    def test_neutral_and_zero_strength(self):
        model = fit(self.rgb, self.rgb, self.masks, self.masks)
        self.assertTrue(np.array_equal(apply(self.rgb, self.masks, model), self.rgb))
        reference = np.clip(self.rgb.astype(float) + 20, 0, 255).astype(np.uint8)
        model = fit(self.rgb, reference, self.masks, self.masks)
        self.assertTrue(np.array_equal(apply(self.rgb, self.masks, model, 0), self.rgb))

    def test_local_brightness_and_saturation(self):
        x = self.rgb.astype(float)
        gray = (x @ [.299, .587, .114])[..., None]
        reference = x.copy()
        reference[:, :24] = gray[:, :24] + (x[:, :24] - gray[:, :24]) * .7 + 12
        reference[:, 24:] -= 12
        reference = np.rint(reference).astype(np.uint8)
        model = fit(self.rgb, reference, self.masks, self.masks)
        result = apply(self.rgb, self.masks, model)
        self.assertLess(np.abs(result.astype(float) - reference).mean(),
                        np.abs(x - reference).mean())

    def test_missing_and_uncertain_regions_fall_back(self):
        reference = np.clip(self.rgb.astype(float) + 20, 0, 255).astype(np.uint8)
        model = fit(self.rgb, reference, self.masks, np.zeros_like(self.masks))
        self.assertTrue(np.array_equal(apply(self.rgb, self.masks, model), self.rgb))
        model = fit(self.rgb, reference, self.masks, self.masks)
        uncertain = np.full_like(self.masks, .5)
        self.assertTrue(np.array_equal(apply(self.rgb, uncertain, model), self.rgb))

    def test_reject_invalid_masks_and_strength(self):
        bad = self.masks.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): fit(self.rgb, self.rgb, bad, self.masks)
        with self.assertRaises(ValueError): fit(self.rgb, self.rgb, self.masks * 2, self.masks)
        with self.assertRaises(ValueError): fit(self.rgb, self.rgb, self.masks[:, :, :1], self.masks)
        model = fit(self.rgb, self.rgb, self.masks, self.masks)
        with self.assertRaises(ValueError): apply(self.rgb, self.masks, model, True)

    def test_one_matching_region_cannot_justify_local_relationship(self):
        masks = np.zeros_like(self.masks)
        masks[:, :, 0] = 1
        reference = (self.rgb.astype(float) + 20).astype(np.uint8)
        model = fit(self.rgb, reference, masks, masks)
        self.assertTrue(np.array_equal(apply(self.rgb, masks, model), self.rgb))


if __name__ == '__main__': unittest.main()
