import unittest
import importlib.util
import importlib
import torch


class TrainingChecks(unittest.TestCase):
    def test_raw_nonfinite_rejected_before_clamp(self):
        self.assertIsNotNone(importlib.util.find_spec('training_checks'))
        api=importlib.import_module('training_checks')
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(ValueError):api.finite_output(torch.full((1,3,8,8),value),(1,3,8,8))
        with self.assertRaises(ValueError):api.finite_output(torch.zeros(1,3,8,8),(1,3,9,8))
        x=torch.full((1,3,8,8),2.)
        self.assertIs(api.finite_output(x,x.shape),x)


if __name__=='__main__':unittest.main()
