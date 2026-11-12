"""Dependency-light input contract; actual model inference is verified separately."""
import unittest
import numpy as np
from canonical_model import validate_rgb


class CanonicalContract(unittest.TestCase):
    def test_rgb8_only(self):
        validate_rgb(np.zeros((1,1,3),np.uint8))
        for value in (np.zeros((0,2,3),np.uint8),np.zeros((2,2,4),np.uint8),
                      np.zeros((2,2,3),float),[[1,2,3]]):
            with self.assertRaises(ValueError):validate_rgb(value)


if __name__=='__main__':unittest.main()
