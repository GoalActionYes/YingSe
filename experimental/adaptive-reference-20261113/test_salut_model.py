import unittest
from pathlib import Path
import ast
import torch
from salut_model import adapted_model_source, filter_inference_state, load_inference_weights


class ModelContractTests(unittest.TestCase):
    def test_pinned_adapter_removes_unsafe_constructor_load(self):
        root = Path(__file__).resolve().parents[2] / '.tools/SA-LUT-source/SA-LUT'
        source = adapted_model_source(root)
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.load' for n in ast.walk(tree)))
        self.assertIn('efficient_attention(Q_flat, K_flat, V_flat, self.attn_temperature)', source)

    def test_missing_encoder_fails_instead_of_using_random_weights(self):
        with self.assertRaises(ValueError):
            filter_inference_state({'vlog2stylenet.CLUTs.LUTs': 1})

    def test_unexpected_model_prefix_fails(self):
        with self.assertRaises(ValueError):
            filter_inference_state({'other_model.weight': 1})

    def test_only_deterministic_training_regularizer_buffer_may_be_absent(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2))
        model.tvmn = torch.nn.Module()
        model.tvmn.register_buffer('weight_c', torch.ones(3)*2)
        restored = load_inference_weights(model, {'weight': torch.zeros(2)})
        self.assertEqual(restored, ['tvmn.weight_c'])
        torch.testing.assert_close(model.weight, torch.zeros(2))
        with self.assertRaises(ValueError):
            load_inference_weights(model, {})


if __name__ == '__main__':
    unittest.main()
