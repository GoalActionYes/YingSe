import unittest
from types import SimpleNamespace
import numpy as np
import torch
from spatial_render import PreparedSpatial


class ByteRenderTests(unittest.TestCase):
    def prepared(self, invalid=False):
        sampler=SimpleNamespace(L_weights=torch.zeros(1,3,1,1),pre_guide=lambda c:c)
        model=SimpleNamespace(sampler=sampler,apply_coeffs=lambda coeff,c: c*float('nan') if invalid else (c*.9+.03).clamp(0,1))
        engine=SimpleNamespace(model=model,tensor=lambda rgb:torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255)
        return PreparedSpatial(engine,torch.ones(1,13,8,16,16),{})

    def test_byte_output_matches_float_oracle_across_bands(self):
        rgb=np.random.default_rng(51).integers(0,256,(37,61,3),dtype=np.uint8);p=self.prepared()
        oracle=np.rint(np.clip(p.float_output(rgb,17),0,1)*255).astype(np.uint8)
        for band in [1,17,128,256]:np.testing.assert_array_equal(p.byte_output(rgb,band),oracle)

    def test_nonfinite_rejected_before_byte_quantization(self):
        with self.assertRaises(ValueError):self.prepared(True).byte_output(np.zeros((7,9,3),np.uint8))

    def test_invalid_band_rejected(self):
        for band in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):self.prepared().byte_output(np.zeros((7,9,3),np.uint8),band)
