import importlib
import unittest
import numpy as np


class ClippingDiagnosticTests(unittest.TestCase):
    def api(self):return importlib.import_module('clipping_diagnostic')
    def test_only_strict_outside_values_have_blocked_clamp_gradient(self):
        raw=np.array([[[-.5,.5,1.5]]]);truth=np.array([[[.5,.5,.5]]]);r=self.api().clipping_statistics(raw,truth)
        self.assertAlmostEqual(r['outside_fraction'],2/3);self.assertEqual(r['outside_error_fraction'],1);self.assertAlmostEqual(r['clipped_l1'],1/3);self.assertAlmostEqual(r['raw_l1'],2/3)
        r=self.api().clipping_statistics(np.array([[[0.,.5,1.]]]),np.array([[[0.,.5,1.]]]))
        self.assertEqual(r['outside_fraction'],0);self.assertEqual(r['outside_error_fraction'],0)
    def test_channel_errors_are_counted_with_same_denominator(self):
        r=self.api().clipping_statistics(np.array([[[-1.,.1,.1]]]),np.array([[[.5,.2,.2]]]))
        self.assertAlmostEqual(r['outside_error_fraction'],.5/.7)
    def test_invalid_shapes_nan_and_truth_range_stop(self):
        api=self.api();a=np.zeros((1,1,3))
        for bad in [np.full((1,1,3),np.nan),np.zeros((0,1,3)),np.zeros((1,1,4))]:
            with self.assertRaises(ValueError):api.clipping_statistics(bad,a)
        with self.assertRaises(ValueError):api.clipping_statistics(a,a+2)


if __name__=='__main__':unittest.main()
