import unittest
import importlib
import numpy as np


class BackgroundInterventionTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('background_intervention')

    def test_person_and_five_pixel_guard_are_never_edited(self):
        api = self.api()
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), bool); mask[20:40, 20:40] = True
        variants = api.make_variants(rgb, mask)
        self.assertEqual(len(variants), 12)
        for name, pixels, info in variants:
            if name.startswith('background'):
                np.testing.assert_array_equal(pixels[15:45, 15:45], rgb[15:45, 15:45])
                self.assertGreater(np.abs(pixels.astype(float)-rgb).mean(), 0)
                self.assertEqual(info['person_max_change'], 0)
        np.testing.assert_array_equal(rgb, np.full((64, 64, 3), 128, dtype=np.uint8))

    def test_coverage_increases_and_opposite_signs_reverse_color(self):
        api=self.api();rgb=np.full((64,64,3),128,np.uint8);mask=np.zeros((64,64),bool)
        variants={name:(pixels,info) for name,pixels,info in api.make_variants(rgb,mask)}
        previous=0
        for coverage in (25,50,100):
            positive,info=variants[f'background_green_{coverage}']
            negative,_=variants[f'background_magenta_{coverage}']
            self.assertGreater(info['mean_edit_weight'],previous);previous=info['mean_edit_weight']
            self.assertLess(float((positive.astype(float)-rgb)[...,0].mean()),0)
            self.assertGreater(float((negative.astype(float)-rgb)[...,0].mean()),0)
            self.assertLess(abs(info['luminance_mean_change']),.5)
        self.assertEqual(previous,1.)

    def test_no_background_is_explicit_and_no_nan(self):
        api=self.api();rgb=np.full((32,32,3),128,np.uint8)
        for name,pixels,info in api.make_variants(rgb,np.ones((32,32),bool)):
            if name.startswith('background'):
                np.testing.assert_array_equal(pixels,rgb)
                self.assertEqual(info['editable_pixels'],0)
                self.assertEqual(info['mean_edit_weight'],0)

    def test_statistics_preserve_direction_and_empty_regions(self):
        api=self.api();base=np.full((2,2,3),100,np.uint8);actual=base.copy();actual[0,0]=[110,95,100]
        mask=np.zeros((2,2),bool);mask[0,0]=True
        row=api.region_delta(actual,base,mask)
        self.assertEqual(row['pixels'],1)
        self.assertEqual(row['rgb_mean_delta'],[10.,-5.,0.])
        self.assertEqual(row['rgb_mae'],5.)
        self.assertAlmostEqual(row['luminance_mean_delta'],.055)
        self.assertTrue(row['sensitive_over_one_code'])
        self.assertIsNone(api.region_delta(actual,base,np.zeros((2,2),bool)))

    def test_neutral_proxy_uses_original_rgb_only(self):
        api=self.api();rgb=np.array([[[100,110,105],[10,10,10],[240,240,240],[50,100,50]]],np.uint8)
        regions=api.target_regions(rgb,np.zeros((1,4),bool))
        np.testing.assert_array_equal(regions['neutral_proxy'],[[True,False,False,False]])

    def test_clipping_is_reported_and_identity_is_exact(self):
        api=self.api();rgb=np.full((32,32,3),250,np.uint8);mask=np.zeros((32,32),bool)
        rows=api.make_variants(rgb,mask)
        np.testing.assert_array_equal(rows[0][1],rgb);np.testing.assert_array_equal(rows[1][1],rgb)
        self.assertEqual(rows[0][0],'identity');self.assertEqual(rows[1][0],'identity_repeat')
        green=next(x for x in rows if x[0]=='background_green_100')
        self.assertGreater(green[2]['preclip_channel_fraction'],0)

    def test_invalid_rgb_mask_and_nonfinite_metrics_rejected(self):
        api=self.api();rgb=np.full((32,32,3),128,np.uint8);mask=np.zeros((32,32),bool)
        for image,m in [(rgb.astype(float),mask),(rgb,mask[:1]),(rgb,np.full((32,32),np.nan))]:
            with self.assertRaises(ValueError):api.make_variants(image,m)
        with self.assertRaises(ValueError):api.region_delta(rgb.astype(float)*np.nan,rgb,mask)


if __name__=='__main__':unittest.main()
