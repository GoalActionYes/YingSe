import importlib
import importlib.util
import unittest
import numpy as np


class BlendCapacityTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('blend_capacity'))
        return importlib.import_module('blend_capacity')
    def test_exact_endpoints_and_identity(self):
        api=self.api();a=np.array([[[0,255,127]]],dtype=np.uint8);b=255-a
        self.assertTrue(np.array_equal(api.blend_codes(a,b,0),a));self.assertTrue(np.array_equal(api.blend_codes(a,b,100),b))
        self.assertTrue(np.array_equal(api.blend_codes(a,a,37),a))
    def test_half_ties_round_to_even(self):
        api=self.api();a=np.array([[[0,1,254]]],dtype=np.uint8);b=a+1
        self.assertTrue(np.array_equal(api.blend_codes(a,b,50),np.array([[[0,2,254]]],dtype=np.uint8)))
    def test_known_optimum_and_earliest_plateau(self):
        api=self.api();a=np.zeros((2,2,3),dtype=np.uint8);b=np.full_like(a,100);gt=np.full_like(a,40)
        scores,step=api.rgb_oracle(a,b,gt);self.assertEqual(len(scores),101);self.assertEqual(step,40);self.assertEqual(scores[40],0)
        scores,step=api.rgb_oracle(a,np.full_like(a,2),np.full_like(a,1));self.assertEqual(step,26)
        scores,step=api.rgb_oracle(a,a,a);self.assertEqual(step,0)
    def test_mae_does_not_wrap_and_invalid_inputs_rejected(self):
        api=self.api();a=np.zeros((1,1,3),dtype=np.uint8);b=np.full_like(a,255)
        scores,_=api.rgb_oracle(a,a,b);self.assertEqual(scores[0],255)
        for step in [-1,101,True,.5]:
            with self.assertRaises(ValueError):api.blend_codes(a,b,step)
        for bad in [b.astype(float),b[:,:,:2],np.empty((0,1,3),dtype=np.uint8)]:
            with self.assertRaises(ValueError):api.blend_codes(a,bad,50)


if __name__=='__main__':unittest.main()
