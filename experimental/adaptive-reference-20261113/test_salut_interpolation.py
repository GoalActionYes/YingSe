"""Independent 16-corner oracle for the pinned author's CUDA forward layout."""
import itertools
import unittest
import numpy as np
import torch
from salut_interpolation import quadrilinear, efficient_attention


def oracle(lut, image):
    dim = lut.shape[-1]
    result = np.zeros((image.shape[0], 3, *image.shape[2:]), np.float64)
    for n, y, x in itertools.product(range(image.shape[0]), range(image.shape[2]), range(image.shape[3])):
        ctx = float(image[n, 0, y, x])
        rgb = image[n, 1:, y, x].astype(np.float64) / (1.000001 / (dim - 1))
        lo = np.floor(rgb).astype(int)
        f = rgb - lo
        for c, r, g, b in itertools.product((0, 1), repeat=4):
            weight = (ctx if c else 1-ctx)
            for channel, bit in enumerate((r, g, b)):
                weight *= f[channel] if bit else 1-f[channel]
            result[n, :, y, x] += weight * lut[:, c, lo[2]+b, lo[1]+g, lo[0]+r]
    return result


class InterpolationTests(unittest.TestCase):
    def test_matches_random_scalar_cuda_oracle_at_boundaries(self):
        rng = np.random.default_rng(719)
        lut = rng.uniform(-.3, 1.4, (3, 2, 5, 5, 5)).astype('float32')
        image = rng.uniform(0, 1, (2, 4, 3, 7)).astype('float32')
        image[:, :, 0, 0] = 0
        image[:, :, 0, -1] = 1
        actual = quadrilinear(torch.from_numpy(lut), torch.from_numpy(image)).numpy()
        np.testing.assert_allclose(actual, oracle(lut, image), atol=6e-7, rtol=0)

    def test_same_rgb_can_have_different_spatial_outputs(self):
        lut = torch.zeros(3, 2, 3, 3, 3)
        lut[:, 1] = torch.tensor([.2, .5, .9])[:, None, None, None]
        image = torch.full((1, 4, 1, 3), .4)
        image[:, 0] = torch.tensor([0., .5, 1.])
        out = quadrilinear(lut, image)
        torch.testing.assert_close(out[0, :, 0, 1], torch.tensor([.1, .25, .45]))
        torch.testing.assert_close(out[0, :, 0, 2], torch.tensor([.2, .5, .9]))
        torch.testing.assert_close(out[0, :, 0, 0], torch.zeros(3))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_scalar_oracle_and_chunked_attention(self):
        rng = np.random.default_rng(392)
        lut = rng.uniform(0, 1, (3, 2, 17, 17, 17)).astype('float32')
        image = rng.uniform(0, 1, (1, 4, 5, 9)).astype('float32')
        image[:, :, 0, :2] = np.array([0., 1.])
        actual = quadrilinear(torch.from_numpy(lut).cuda(), torch.from_numpy(image).cuda()).cpu().numpy()
        np.testing.assert_allclose(actual, oracle(lut, image), atol=3e-6, rtol=0)
        q = torch.randn(1, 19, 16, device='cuda')
        k = torch.randn(1, 16, 23, device='cuda')
        v = torch.randn(1, 23, 16, device='cuda')
        temperature = torch.tensor(1.23, device='cuda')
        expected = torch.softmax(q@k/(16**.5)*temperature, dim=-1)@v
        torch.testing.assert_close(efficient_attention(q, k, v, temperature, chunk_rows=3), expected, atol=3e-6, rtol=1e-5)

    def test_bad_context_or_shapes_rejected(self):
        lut = torch.zeros(3, 2, 3, 3, 3)
        with self.assertRaises(ValueError):
            quadrilinear(lut, torch.full((1, 4, 1, 1), float('nan')))
        with self.assertRaises(ValueError):
            quadrilinear(lut, torch.full((1, 4, 1, 1), 1.1))
        with self.assertRaises(ValueError):
            quadrilinear(torch.zeros(3, 3, 3, 3, 3), torch.zeros(1, 4, 1, 1))

    def test_attention_matches_author_formula_and_temperature(self):
        generator = torch.Generator().manual_seed(48)
        q = torch.randn(2, 13, 8, generator=generator)
        k = torch.randn(2, 8, 17, generator=generator)
        v = torch.randn(2, 17, 8, generator=generator)
        temperature = torch.tensor(1.37)
        expected = torch.softmax(torch.bmm(q, k) / (8**.5) * temperature, dim=-1) @ v
        torch.testing.assert_close(efficient_attention(q, k, v, temperature), expected, atol=5e-7, rtol=1e-6)
        torch.testing.assert_close(efficient_attention(q, k, v, temperature, chunk_rows=2), expected, atol=5e-7, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
