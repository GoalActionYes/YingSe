import unittest
import numpy as np
from semantic_crops import choose_crops,blend_residual


class CropChecks(unittest.TestCase):
    def test_background_only_and_flat_regions_rejected(self):
        rgb=np.full((100,100,3),128,np.uint8)
        self.assertFalse(choose_crops(rgb,np.zeros((100,100)))['accepted'])
        person=np.zeros((100,100));person[20:80,30:70]=1
        self.assertFalse(choose_crops(rgb,person)['accepted'])

    def test_regions_and_uniform_residual_preserve_global(self):
        rgb=np.random.default_rng(2).integers(40,200,(160,160,3),dtype=np.uint8)
        person=np.zeros((160,160));person[40:120,60:100]=1
        self.assertTrue(choose_crops(rgb,person)['accepted'])
        global_rgb=np.full_like(rgb,128);same_shift=np.full_like(rgb,140)
        result=blend_residual(global_rgb,same_shift,same_shift,person)
        np.testing.assert_array_equal(result,global_rgb)

    def test_opposed_regions_and_uncertain_edges(self):
        base=np.full((100,100,3),128,np.uint8)
        p=np.zeros((100,100));p[:,:50]=1
        result=blend_residual(base,np.full_like(base,148),np.full_like(base,108),p)
        self.assertGreater(result[:,10].mean(),128)
        self.assertLess(result[:,90].mean(),128)
        uncertain=np.full((100,100),.5)
        np.testing.assert_array_equal(blend_residual(base,base+10,base-10,uncertain),base)


if __name__=='__main__':unittest.main()
