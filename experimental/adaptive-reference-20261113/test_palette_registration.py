import importlib
import importlib.util
import unittest
import numpy as np


class RegistrationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('palette_registration'))
        return importlib.import_module('palette_registration')

    def test_identity_with_permuted_palette(self):
        api=self.api();x=np.random.default_rng(27).uniform(.1,.9,(40,3))
        matrix,info=api.register(x,x[::-1])
        np.testing.assert_allclose(np.c_[x,np.ones(len(x))]@matrix,x,atol=1e-5)

    def test_small_known_cast_is_recovered_without_correspondence_labels(self):
        api=self.api();x=np.random.default_rng(28).uniform(.1,.9,(30,3))
        y=x*[1.02,.98,1.01]+[.01,.008,-.006]
        matrix,info=api.register(x,y[::-1])
        self.assertLess(np.abs(np.c_[x,np.ones(len(x))]@matrix-y).mean(),.003)

    def test_nonfinite_or_empty_input_is_rejected(self):
        api=self.api()
        for x in [np.ones((0,3)),np.full((8,3),np.nan)]:
            with self.assertRaises(ValueError):api.register(x,np.ones((8,3)))


if __name__=='__main__':unittest.main()
