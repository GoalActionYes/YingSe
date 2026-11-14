import unittest
import numpy as np


class SelfBiasTests(unittest.TestCase):
    def test_self_reference_is_exact(self):
        from self_bias import cancel
        x=np.random.default_rng(51).integers(0,256,(7,11,3),dtype=np.uint8)
        raw=x.astype(np.float32)/255+.08
        np.testing.assert_array_equal(cancel(x,raw,raw),x)

    def test_known_bias_cancels_without_losing_grade(self):
        from self_bias import cancel
        x=np.full((7,11,3),100,np.uint8);saved=x.copy();base=x.astype(np.float32)/255
        bias=np.array([.04,-.03,.08],np.float32);grade=np.array([.1,.02,-.1],np.float32)
        raw=base+bias+grade;raw_saved=raw.copy()
        expected=np.rint(np.clip(base+grade,0,1)*255).astype(np.uint8)
        np.testing.assert_array_equal(cancel(x,raw,base+bias),expected)
        np.testing.assert_array_equal(x,saved);np.testing.assert_array_equal(raw,raw_saved)

    def test_invalid_values_fail_closed(self):
        from self_bias import cancel
        x=np.zeros((7,11,3),np.uint8);a=x.astype(np.float32)
        for raw,self_raw in [(a[:3],a),(a,a[:,:,:2]),(a+float('nan'),a),(a,a+float('inf'))]:
            with self.assertRaises(ValueError):cancel(x,raw,self_raw)
        with self.assertRaises(ValueError):cancel(x.astype(float),a,a)
