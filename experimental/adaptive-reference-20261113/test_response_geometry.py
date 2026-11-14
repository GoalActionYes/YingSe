import importlib
import unittest
import numpy as np


class ResponseGeometryTests(unittest.TestCase):
    def api(self):return importlib.import_module('response_geometry')
    def frame(self,change):return (np.full((2,2,3),100)+np.asarray(change)).astype(np.uint8)
    def test_exact_ignored_reversed_and_scaled_changes(self):
        api=self.api();base=self.frame([0,0,0]);gt=self.frame([10,0,0])
        for change,gain,amp,cos in [([10,0,0],1,1,1),([0,0,0],0,0,None),([-10,0,0],-1,1,-1),([20,0,0],2,2,1)]:
            r=api.response_geometry(base,self.frame(change),base,gt)
            self.assertAlmostEqual(r['gain'],gain);self.assertAlmostEqual(r['amplitude_ratio'],amp)
            if cos is None:self.assertIsNone(r['cosine'])
            else:self.assertAlmostEqual(r['cosine'],cos)
            self.assertAlmostEqual(r['orthogonal_ratio'],0)
    def test_orthogonal_change_is_not_merely_weak_response(self):
        base=self.frame([0,0,0]);r=self.api().response_geometry(base,self.frame([0,10,0]),base,self.frame([10,0,0]))
        self.assertEqual(r['gain'],0);self.assertEqual(r['cosine'],0);self.assertEqual(r['amplitude_ratio'],1);self.assertEqual(r['orthogonal_ratio'],1)
    def test_zero_expected_and_invalid_frames(self):
        api=self.api();base=self.frame([0,0,0]);r=api.response_geometry(base,self.frame([10,0,0]),base,base)
        self.assertTrue(all(v is None for v in r.values()))
        for bad in [base.astype(float),base[:0],np.zeros((2,2,4),dtype=np.uint8)]:
            with self.assertRaises(ValueError):api.response_geometry(base,bad,base,base)


if __name__=='__main__':unittest.main()
