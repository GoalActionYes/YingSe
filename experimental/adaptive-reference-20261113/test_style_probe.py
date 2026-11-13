import unittest
import torch
from style_probe import render_style, draw_style_codes, masked_parameter_mse, StyleRegressor


class StyleProbeTests(unittest.TestCase):
    def test_zero_style_preserves_pixels_and_shape(self):
        rgb = torch.rand(3, 3, 11, 17, generator=torch.Generator().manual_seed(4))
        mask = torch.rand(3, 1, 11, 17, generator=torch.Generator().manual_seed(9))
        actual = render_style(rgb, mask, torch.zeros(3, 2, 6))
        torch.testing.assert_close(actual, rgb, atol=1e-7, rtol=0)

    def test_equal_colors_in_different_regions_can_receive_different_styles(self):
        rgb = torch.full((1, 3, 2, 4), .5)
        mask = torch.zeros(1, 1, 2, 4)
        mask[:, :, :, :2] = 1
        code = torch.zeros(1, 2, 6)
        code[:, 0, 3] = 1
        code[:, 1, 3] = -1
        actual = render_style(rgb, mask, code)
        self.assertGreater(float(actual[:, :, :, :2].mean()), .54)
        self.assertLess(float(actual[:, :, :, 2:].mean()), .46)
        self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))

    def test_style_labels_reused_for_cross_content_pairs(self):
        codes = draw_style_codes(12, torch.Generator().manual_seed(105))
        paired = codes.repeat_interleave(2, dim=0)
        torch.testing.assert_close(paired[::2], paired[1::2])
        self.assertEqual(tuple(codes.shape), (12, 2, 6))
        self.assertTrue(bool((codes.abs() <= 1).all()))

    def test_unobservable_person_style_does_not_change_loss(self):
        truth = torch.zeros(2, 2, 6)
        prediction = torch.zeros_like(truth)
        prediction[:, 0] = 10
        availability = torch.tensor([[0., 1.], [0., 1.]])
        self.assertEqual(float(masked_parameter_mse(prediction, truth, availability)), 0.)
        availability[0, 0] = 1
        self.assertGreater(float(masked_parameter_mse(prediction, truth, availability)), 0.)

    def test_learned_predictor_has_bounded_output_and_gradients(self):
        model = StyleRegressor(42, 120, use_semantic=True)
        prediction = model(torch.randn(4, 42), torch.randn(4, 120))
        self.assertEqual(tuple(prediction.shape), (4, 2, 6))
        self.assertTrue(bool((prediction.abs() <= 1).all()))
        loss = masked_parameter_mse(prediction, torch.ones_like(prediction)*.4, torch.ones(4, 2))
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
