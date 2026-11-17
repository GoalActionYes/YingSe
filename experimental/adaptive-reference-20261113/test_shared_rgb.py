import unittest
import numpy as np
from shared_rgb import analyze


class SharedRGBTests(unittest.TestCase):
    def test_same_rgb_conflicting_outputs_has_exact_floor(self):
        x=np.full((2,2,3),80,np.uint8);a=np.zeros_like(x);b=np.full_like(x,100)
        r=analyze(x,a,x,b)
        self.assertEqual(r['joint_minimum_mae'],50)
        self.assertEqual(r['cross_image_excess_minimum_mae'],50)
        self.assertEqual(r['shared_key_median_output_gap'],100)
        self.assertEqual(r['target_shared_pixel_fraction'],1)

    def test_no_shared_inputs_has_no_cross_image_cost(self):
        a=np.zeros((2,2,3),np.uint8);b=np.ones_like(a)
        r=analyze(a,a,b,np.full_like(b,100))
        self.assertEqual(r['shared_rgb_keys'],0)
        self.assertEqual(r['cross_image_excess_minimum_mae'],0)
        self.assertIsNone(r['shared_key_median_output_gap'])

    def test_within_image_conflict_is_not_attributed_to_cross_image(self):
        x=np.zeros((1,2,3),np.uint8);y=np.array([[[0]*3,[100]*3]],np.uint8)
        r=analyze(x,y,x,y)
        self.assertEqual(r['joint_minimum_mae'],50)
        self.assertEqual(r['cross_image_excess_minimum_mae'],0)

    def test_small_random_case_matches_brute_force_group_medians(self):
        rng=np.random.default_rng(13);a=rng.integers(0,3,(3,4,3),dtype=np.uint8);b=rng.integers(0,3,(3,4,3),dtype=np.uint8)
        y=rng.integers(0,256,a.shape,dtype=np.uint8);z=rng.integers(0,256,b.shape,dtype=np.uint8)
        groups={}
        for rgb,value in zip(np.concatenate([a.reshape(-1,3),b.reshape(-1,3)]),np.concatenate([y.reshape(-1,3),z.reshape(-1,3)])):groups.setdefault(tuple(rgb),[]).append(value.astype(float))
        total=0
        for values in groups.values():total+=np.abs(np.array(values)-np.median(values,axis=0)).sum()
        self.assertEqual(analyze(a,y,b,z)['joint_minimum_mae'],total/(a.size+b.size))
        with self.assertRaises(ValueError):analyze(a.astype(float),y,b,z)
