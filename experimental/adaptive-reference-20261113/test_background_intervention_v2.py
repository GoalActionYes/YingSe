import importlib
import unittest
import numpy as np


class ModelLuminanceTests(unittest.TestCase):
    def api(self):return importlib.import_module('background_intervention_v2')

    def test_endpoints_and_saturated_colors_remain_unchanged(self):
        api=self.api();rgb=np.array([[[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255]]],np.uint8)
        raw,info=api.chroma_shift(rgb,np.ones((1,5)),1)
        np.testing.assert_array_equal(raw,rgb)
        self.assertEqual(info['symmetric_scale_zero_fraction'],1.)

    def test_symmetric_gamut_and_model_y_bound_on_random_pixels(self):
        api=self.api();rgb=np.random.default_rng(1611).integers(0,256,(64,64,3),dtype=np.uint8)
        w=np.random.default_rng(4).random((64,64));a,info=api.chroma_shift(rgb,w,1);b,_=api.chroma_shift(rgb,w,-1)
        np.testing.assert_allclose(a+b,2*rgb.astype(float),atol=1e-12)
        for raw in (a,b):
            self.assertGreaterEqual(raw.min(),0);self.assertLessEqual(raw.max(),255)
            delta=raw-rgb.astype(float)
            self.assertLess(np.abs(delta@np.array([.2126,.7152,.0722])).max(),1e-10)
            q=np.rint(raw).astype(np.uint8).astype(float)-rgb
            self.assertLessEqual(np.abs(q@np.array([.2126,.7152,.0722])).max(),.500001)

    def test_variants_protect_person_and_have_model_y_statistics(self):
        api=self.api();rgb=np.full((64,64,3),128,np.uint8);mask=np.zeros((64,64),bool);mask[20:40,20:40]=True
        rows=api.make_variants(rgb,mask);self.assertEqual(len(rows),12)
        for n,p,info in rows:
            if n.startswith('background'):
                np.testing.assert_array_equal(p[15:45,15:45],rgb[15:45,15:45])
                self.assertEqual(info['preclip_channel_fraction'],0)
                self.assertLessEqual(info['model_y_max_abs_quantized_delta'],.500001)
                self.assertGreater(info['reference_rgb_mae'],0)

    def test_invalid_weights_rejected_before_intervention(self):
        api=self.api();rgb=np.full((8,8,3),128,np.uint8)
        for w in [np.ones((4,8)),np.full((8,8),np.nan),np.full((8,8),-1.),np.full((8,8),2.)]:
            with self.assertRaises(ValueError):api.chroma_shift(rgb,w,1)


if __name__=='__main__':unittest.main()
