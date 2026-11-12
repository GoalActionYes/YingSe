import unittest
import torch
from salut_domain import fit_inverse, apply_inverse, quantize_log_input
from salut_interpolation import trilinear


def identity(dim):
    b, g, r = torch.meshgrid(*[torch.linspace(0, 1, dim)]*3, indexing='ij')
    return torch.stack((r, g, b))


class InverseDomainTests(unittest.TestCase):
    def test_coupled_nonidentity_transform_roundtrip(self):
        cube = identity(9)
        # Color coupling prevents three independent channel inverses from passing.
        matrix = torch.tensor([[.70, .05, .02], [.04, .73, .03], [.02, .06, .70]])
        forward = (matrix @ cube.reshape(3, -1)).reshape_as(cube)+.07
        inv, _ = fit_inverse(forward, dim=13, steps=12)
        log = torch.rand(1, 3, 4, 7, generator=torch.Generator().manual_seed(23))*.7+.15
        rgb = trilinear(forward, log)
        recovered, error = apply_inverse(inv, forward, rgb)
        torch.testing.assert_close(recovered, log, atol=.0004, rtol=0)
        self.assertLess(float(error.max()), .0004)

    def test_unreachable_color_is_reported_not_claimed_exact(self):
        forward = identity(5)*.6+.2
        inv, diagnostics = fit_inverse(forward, dim=9, steps=8)
        _, error = apply_inverse(inv, forward, torch.zeros(1, 3, 1, 1))
        self.assertGreater(float(error.max()), .19)
        self.assertGreater(diagnostics['grid_residual_fraction_over_2_codes'], .1)

    def test_reachable_color_is_not_lost_at_clipped_neutral_initialization(self):
        forward = identity(9)
        forward[0] = (3*forward[0]-2*forward[1]).clamp(0, 1)
        inverse, _ = fit_inverse(forward, dim=9, steps=12)
        rgb = torch.tensor([.25, .875, .5]).reshape(1, 3, 1, 1)
        _, error = apply_inverse(inverse, forward, rgb)
        self.assertLess(float(error.max()), .0001)

    def test_lut_layout_roundtrip_preserves_channels(self):
        forward = identity(5)
        inv, _ = fit_inverse(forward, dim=9, steps=8)
        rgb = torch.tensor([.14, .55, .86]).reshape(1, 3, 1, 1)
        recovered, error = apply_inverse(inv, forward, rgb)
        torch.testing.assert_close(recovered, rgb, atol=2e-6, rtol=0)
        self.assertLess(float(error.max()), 2e-6)

    def test_model_roundtrip_includes_quantization_amplification(self):
        forward = (identity(11)*5-2).clamp(0, 1)
        continuous = torch.full((1, 3, 1, 1), .45294)
        rgb = trilinear(forward, continuous)
        codes, error = quantize_log_input(forward, continuous, rgb)
        self.assertEqual(codes.dtype, torch.uint8)
        expected = (trilinear(forward, codes.float()/255)-rgb).abs()
        torch.testing.assert_close(error, expected)
        self.assertGreater(float(error.max()*255), 2.4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_convex_interpolation_respects_unit_bounds_on_cuda(self):
        inverse = torch.ones(3, 9, 9, 9, device='cuda')
        forward = identity(9).cuda()
        rgb = torch.rand(1, 3, 17, 19, generator=torch.Generator().manual_seed(832)).cuda()
        converted, error = apply_inverse(inverse, forward, rgb)
        self.assertTrue(bool((converted <= 1).all()))
        self.assertTrue(bool(torch.isfinite(error).all()))

    def test_invalid_inverse_is_rejected_before_roundoff_clamping(self):
        with self.assertRaises(ValueError):
            apply_inverse(torch.full((3, 5, 5, 5), 1.1), identity(5), torch.ones(1, 3, 1, 1))


if __name__ == '__main__':
    unittest.main()
