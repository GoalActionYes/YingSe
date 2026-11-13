import unittest
import torch
from style_probe import render_style
from style_composition import compose_styles, fallback_unobserved_roles, _zero_preserving_power


class StyleCompositionTests(unittest.TestCase):
    def test_equal_styles_preserve_every_rgb8_value_without_gamut_projection(self):
        generator = torch.Generator().manual_seed(56)
        rgb = torch.randint(0, 256, (4, 3, 17, 23), generator=generator).float()/255
        person = torch.rand(4, 1, 17, 23, generator=generator)
        code = torch.rand(4, 2, 6, generator=generator)*2-1
        result = compose_styles(rgb, person, code, code)
        torch.testing.assert_close(result.mul(255).round(), rgb.mul(255).round(), atol=0, rtol=0)

    def test_known_unclipped_input_style_is_replaced_by_reference_style(self):
        generator = torch.Generator().manual_seed(84)
        rgb = torch.rand(2, 3, 9, 13, generator=generator)*.3+.3
        person = torch.rand(2, 1, 9, 13, generator=generator)
        source = (torch.rand(2, 2, 6, generator=generator)*2-1)*.3
        desired = (torch.rand(2, 2, 6, generator=generator)*2-1)*.4
        input_rgb = render_style(rgb, person, source)
        actual = compose_styles(input_rgb, person, source, desired)
        expected = render_style(rgb, person, desired)
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=0)

    def test_finite_gradients_at_black_white_and_clipped_colors(self):
        rgb = torch.tensor([0., 1., .02, .98]).reshape(1, 1, 2, 2).expand(1, 3, 2, 2)
        person = torch.tensor([0., 1., .3, .7]).reshape(1, 1, 2, 2)
        source = torch.full((1, 2, 6), .2, requires_grad=True)
        desired = torch.full((1, 2, 6), -.3, requires_grad=True)
        output = compose_styles(rgb, person, source, desired)
        output.mean().backward()
        self.assertTrue(bool(torch.isfinite(source.grad).all()))
        self.assertTrue(bool(torch.isfinite(desired.grad).all()))
        self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))

    def test_positive_gamma_at_exact_black_has_finite_backward(self):
        rgb = torch.zeros(1, 3, 2, 2, requires_grad=True)
        person = torch.ones(1, 1, 2, 2)*.5
        source = torch.zeros(1, 2, 6, requires_grad=True)
        desired = torch.zeros(1, 2, 6)
        desired[:, :, 5] = .2
        desired.requires_grad_(True)
        output = compose_styles(rgb, person, source, desired)
        self.assertTrue(bool((output == 0).all()))
        output.mean().backward()
        self.assertTrue(bool(torch.isfinite(source.grad).all()))
        self.assertTrue(bool(torch.isfinite(desired.grad).all()))
        self.assertTrue(bool(torch.isfinite(rgb.grad).all()))

    def test_finite_power_boundary_preserves_rgb8_forward_codes(self):
        values = torch.cat((torch.zeros(1), torch.logspace(-12, 0, 1000), torch.arange(256)/255))[:, None]
        exponent = torch.linspace(-.15, .15, 31).exp()[None]
        expected = values.pow(exponent).mul(255).round()
        actual = _zero_preserving_power(values, exponent).mul(255).round()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_missing_reference_person_uses_observed_background_style(self):
        code = torch.rand(2, 2, 6)
        probabilities = torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4)*.4))
        result = fallback_unobserved_roles(code, probabilities)
        torch.testing.assert_close(result[0, 0], code[0, 1])
        torch.testing.assert_close(result[1], code[1])


if __name__ == '__main__':
    unittest.main()
