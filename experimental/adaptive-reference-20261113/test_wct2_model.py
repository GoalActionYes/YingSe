import ast
import unittest
import numpy as np
import torch
from wct2_model import adapted_transfer_class, prepare_rgb, restore_rgb


class WCT2ContractTests(unittest.TestCase):
    def test_only_safe_weight_loads_in_adapted_class(self):
        source = adapted_transfer_class()
        calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load']
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(any(k.arg == 'weights_only' and isinstance(k.value, ast.Constant) and k.value.value is True for k in n.keywords) for n in calls))

    def test_padding_preserves_all_pixels_and_dimensions(self):
        rgb = np.random.default_rng(12).integers(0, 256, (37, 51, 3), dtype='uint8')
        tensor, working = prepare_rgb(rgb, device='cpu', max_edge=640)
        self.assertEqual(tuple(tensor.shape), (1, 3, 48, 64))
        np.testing.assert_array_equal(working, rgb)
        np.testing.assert_array_equal(restore_rgb(tensor, working.shape), rgb)

    def test_size_reduction_is_explicit_and_preserves_aspect(self):
        rgb = np.zeros((100, 200, 3), dtype='uint8')
        tensor, working = prepare_rgb(rgb, device='cpu', max_edge=64)
        self.assertEqual(working.shape, (32, 64, 3))
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 64))


if __name__ == '__main__':
    unittest.main()
