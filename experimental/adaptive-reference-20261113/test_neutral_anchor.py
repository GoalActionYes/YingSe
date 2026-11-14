import unittest
import numpy as np


class NeutralAnchorTests(unittest.TestCase):
    def api(self):
        from neutral_anchor import correct_neutral_highlights
        return correct_neutral_highlights

    def test_identity_is_exact(self):
        rgb=np.full((32,32,3),190,np.uint8)
        result,info=self.api()(rgb,rgb,rgb)
        np.testing.assert_array_equal(result,rgb)

    def test_extra_green_is_reduced_without_changing_colored_pixels(self):
        source=np.full((32,32,3),190,np.uint8);source[:8]=[180,40,30]
        reference=source.copy();graded=source.copy();graded[8:]=[180,205,180]
        result,info=self.api()(source,reference,graded)
        self.assertFalse(info['bypass'])
        self.assertLess(float(result[10,10,1]-result[10,10,0]),25)
        np.testing.assert_array_equal(result[:8],graded[:8])

    def test_no_reliable_reference_anchor_preserves_intended_blue(self):
        source=np.full((32,32,3),190,np.uint8);reference=np.full_like(source,[80,130,210]);graded=reference.copy()
        result,info=self.api()(source,reference,graded)
        self.assertTrue(info['bypass']);np.testing.assert_array_equal(result,graded)

    def test_matching_warm_reference_is_not_forced_to_gray(self):
        source=np.full((32,32,3),190,np.uint8);reference=np.full_like(source,[200,190,180])
        result,info=self.api()(source,reference,reference)
        np.testing.assert_array_equal(result,reference)

    def test_delta_bound_and_luminance_projection(self):
        source=np.full((32,32,3),190,np.uint8);graded=np.full_like(source,[120,220,120])
        result,info=self.api()(source,source,graded)
        delta=np.array(info['delta'])
        self.assertLessEqual(float(np.abs(delta).max()),.02000001)
        self.assertLess(abs(float(delta@[.299,.587,.114])),1e-12)

    def test_invalid_input_rejected(self):
        source=np.zeros((32,32,3),np.uint8)
        for invalid in [source.astype(float),source[:,:,:2],source[:0]]:
            with self.assertRaises(ValueError):self.api()(invalid,source,source)
        with self.assertRaises(ValueError):self.api()(source,source,source[:16])
