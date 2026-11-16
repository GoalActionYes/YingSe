import unittest
import importlib
import numpy as np


class HistogramReplayTests(unittest.TestCase):
    def test_conditions_isolate_channels_without_mutating_sources(self):
        api=importlib.import_module('histogram_replay')
        a=np.full((1,3,128),1/128,np.float32);b=np.zeros_like(a);b[:,:,3]=1
        before=a.copy();rows=api.conditions(a,b)
        self.assertEqual(set(rows),{'full','y_only','chroma_only'})
        np.testing.assert_array_equal(rows['y_only'][:,0],b[:,0]);np.testing.assert_array_equal(rows['y_only'][:,1:],a[:,1:])
        np.testing.assert_array_equal(rows['chroma_only'][:,0],a[:,0]);np.testing.assert_array_equal(rows['chroma_only'][:,1:],b[:,1:])
        rows['full'][:]=0
        np.testing.assert_array_equal(a,before);self.assertEqual(b.sum(),3.)

    def test_invalid_nonfinite_unnormalized_and_negative_histograms_rejected(self):
        api=importlib.import_module('histogram_replay');a=np.full((1,3,128),1/128,np.float32)
        for b in [a[:,:,:2],a*np.nan,a*2,-a]:
            with self.assertRaises(ValueError):api.conditions(a,b)


if __name__=='__main__':unittest.main()
