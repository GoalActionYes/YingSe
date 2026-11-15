import unittest
import numpy as np


class SensitivityBasisTests(unittest.TestCase):
    def test_known_active_direction_and_rotation(self):
        from condition_subspace import sensitivity_basis
        g = np.array([[3., 0, 0], [-3, 0, 0], [0, 1, 0], [0, -1, 0]])
        values, basis = sensitivity_basis(g)
        np.testing.assert_allclose(values, [4.5, .5, 0], atol=1e-12)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(basis[:, 0], [1, 0, 0], atol=1e-12)
        q, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(3, 3)))
        other_values, other_basis = sensitivity_basis(g @ q)
        np.testing.assert_allclose(values, other_values, atol=1e-12)
        np.testing.assert_allclose(np.outer(other_basis[:, 0], other_basis[:, 0]), q.T @ np.diag([1., 0, 0]) @ q, atol=1e-12)

    def test_invalid_or_no_sensitivity(self):
        from condition_subspace import sensitivity_basis
        for x in [np.zeros((4, 3)), np.array([1., 2]), np.ones((0, 3)), np.full((4, 3), np.nan)]:
            with self.assertRaises(ValueError): sensitivity_basis(x)
