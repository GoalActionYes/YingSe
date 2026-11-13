import unittest
import torch
from global_grid import collapse_grid
from test_capacity_audit import TinyModel


class GlobalGridTests(unittest.TestCase):
    def test_projection_averages_premultiplied_channels_and_keeps_positive_alpha(self):
        grid=torch.randn(2,13,8,16,16);grid[:,12]=torch.rand(2,8,16,16)+.1
        projected=collapse_grid(grid);self.assertEqual(tuple(projected.shape),(2,13,8,1,1));torch.testing.assert_close(projected,grid.mean((-2,-1),keepdim=True));self.assertTrue(bool((projected[:,12]>0).all()))
        # mean(numerator)/mean(alpha), not mean(numerator/alpha).
        self.assertFalse(torch.allclose(projected[:,:12]/projected[:,12:],(grid[:,:12]/grid[:,12:]).mean((-2,-1),keepdim=True)))

    def test_fixed_projected_grid_is_position_independent_and_pixel_permutation_equivariant(self):
        model=TinyModel().eval();grid=torch.randn(1,13,8,16,16)*.1;grid[:,12]=1;projected=collapse_grid(grid)
        image=torch.rand(1,3,7,9);image[:,:,0,0]=image[:,:,5,6]
        out=model.apply_coeffs(model.sampler(projected,image),image)
        torch.testing.assert_close(out[:,:,0,0],out[:,:,5,6],atol=0,rtol=0)
        order=torch.randperm(63);other=image.flatten(2)[:,:,order].reshape_as(image);actual=model.apply_coeffs(model.sampler(projected,other),other)
        torch.testing.assert_close(actual,out.flatten(2)[:,:,order].reshape_as(out),atol=1e-7,rtol=0)

    def test_projection_has_finite_uniform_spatial_gradients(self):
        grid=torch.ones(1,13,8,16,16,requires_grad=True);collapse_grid(grid).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(grid.grad).all()));self.assertGreater(float(grid.grad.abs().sum()),0)
        torch.testing.assert_close(grid.grad,grid.grad[:,:,:,:1,:1].expand_as(grid.grad))

    def test_invalid_shapes_alpha_and_nonfinite_values_fail(self):
        for grid in [torch.zeros(1,13,8,16,16),torch.ones(1,12,8,16,16),torch.full((1,13,8,16,16),float('nan'))]:
            with self.assertRaises(ValueError):collapse_grid(grid)


if __name__=='__main__':unittest.main()
