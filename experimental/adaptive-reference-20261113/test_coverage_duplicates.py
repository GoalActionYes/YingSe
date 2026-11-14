import importlib
import unittest
import numpy as np
from PIL import Image


class CoverageDuplicateTests(unittest.TestCase):
    def api(self):return importlib.import_module('coverage_duplicates')
    def image(self):
        rng=np.random.default_rng(42);a=rng.integers(50,150,(12,15),dtype=np.uint8)
        return np.repeat(np.array(Image.fromarray(a).resize((240,190),Image.Resampling.BILINEAR))[...,None],3,axis=2)
    def test_same_picture_resize_and_positive_luma_affine_remain_close(self):
        api=self.api();a=self.image();h=api.perceptual_hash(a)
        self.assertEqual(h,api.perceptual_hash(a.copy()))
        resized=np.array(Image.fromarray(a).resize((480,380),Image.Resampling.BILINEAR))
        changed=np.rint(a.astype(float)*.7+25).astype(np.uint8)
        self.assertLessEqual((h^api.perceptual_hash(resized)).bit_count(),6)
        self.assertLessEqual((h^api.perceptual_hash(changed)).bit_count(),6)
    def test_unrelated_spatial_pattern_not_classed_as_same(self):
        api=self.api();a=self.image();b=np.roll(a,80,axis=1)
        self.assertGreater((api.perceptual_hash(a)^api.perceptual_hash(b)).bit_count(),6)
    def test_flat_is_finite_and_invalid_input_is_rejected(self):
        api=self.api()
        self.assertEqual(api.perceptual_hash(np.full((32,32,3),100,dtype=np.uint8)),0)
        for a in [np.zeros((3,3)),np.zeros((32,32,3)),np.zeros((0,32,3),dtype=np.uint8)]:
            with self.assertRaises(ValueError):api.perceptual_hash(a)
    def test_exact_hash_includes_dimensions_and_near_flags_keep_all_pairs(self):
        api=self.api();a=self.image()
        self.assertNotEqual(api.pixel_digest(a),api.pixel_digest(a.reshape(240,190,3)))
        records=[dict(path='a',hash=0),dict(path='b',hash=1)]
        held=[dict(path='c',hash=0),dict(path='d',hash=2**63-1)]
        self.assertEqual(api.near_crossings(records,held),[dict(train='a',heldout='c',distance=0),dict(train='b',heldout='c',distance=1)])


if __name__=='__main__':unittest.main()
