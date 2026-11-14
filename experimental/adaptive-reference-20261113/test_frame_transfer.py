import unittest
from types import SimpleNamespace
import numpy as np
import torch
from spatial_render import PreparedSpatial


class FrameTransferTests(unittest.TestCase):
    def prepared(self,kind=None):
        def guide(c):
            return c*float('nan') if kind=='guide' else c
        def coeff(coeff,c):
            if kind=='raw' or (kind=='late' and c.mean()>.5):return c*float('nan')
            return c*.9+.03+coeff[:,:3]*.02
        model=SimpleNamespace(sampler=SimpleNamespace(L_weights=torch.zeros(1,3,1,1),pre_guide=guide),apply_coeffs=coeff)
        engine=SimpleNamespace(model=model,tensor=lambda x:torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255)
        return PreparedSpatial(engine,torch.rand(1,13,8,16,16,generator=torch.Generator().manual_seed(53)),{})

    def test_bytes_match_original_across_shapes_and_bands(self):
        from frame_transfer import render
        rng=np.random.default_rng(135)
        for shape in [(1,1,3),(37,61,3),(257,31,3)]:
            rgb=rng.integers(0,256,shape,dtype=np.uint8);p=self.prepared()
            for band in [1,17,128]:
                # float_output is the independent original path; byte_output
                # delegates to render after integration and is not an oracle.
                expected=np.rint(np.clip(p.float_output(rgb,band),0,1)*255).astype(np.uint8)
                np.testing.assert_array_equal(render(p,rgb,band),expected)
            flipped=rgb[:,::-1];saved=flipped.copy()
            expected=np.rint(np.clip(p.float_output(flipped,128),0,1)*255).astype(np.uint8)
            np.testing.assert_array_equal(render(p,flipped),expected)
            np.testing.assert_array_equal(flipped,saved)

    def test_invalid_guide_raw_and_late_band_never_return_image(self):
        from frame_transfer import render
        rgb=np.zeros((19,11,3),np.uint8);rgb[10:]=255
        for kind in ['guide','raw','late']:
            with self.assertRaises(ValueError):render(self.prepared(kind),rgb,10)

    def test_nonfinite_elements_are_zero_before_cast(self):
        from frame_transfer import finite_or_zero
        value=torch.tensor([1.,float('nan'),float('inf'),-float('inf')])
        safe,valid=finite_or_zero(value,(4,))
        self.assertFalse(valid.item());self.assertTrue(torch.isfinite(safe).all())
        torch.testing.assert_close(safe,torch.tensor([1.,0.,0.,0.]))
        with self.assertRaises(ValueError):finite_or_zero(value,(3,))

    def test_invalid_inputs_fail_closed(self):
        from frame_transfer import render
        rgb=np.zeros((3,4,3),np.uint8)
        for band in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):render(self.prepared(),rgb,band)
        with self.assertRaises(ValueError):render(self.prepared(),rgb[:0])
