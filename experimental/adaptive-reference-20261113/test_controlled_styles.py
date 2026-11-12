import unittest
import numpy as np
from controlled_styles import stylize,PROFILES


class StyleChecks(unittest.TestCase):
    def test_identity_and_background_only(self):
        rgb=np.full((8,8,3),[100,120,140],np.uint8)
        zero=np.zeros((8,8))
        np.testing.assert_array_equal(stylize(rgb,zero,'identity'),rgb)
        a=stylize(rgb,zero,'region_color_luminance')
        b=stylize(rgb,np.ones((8,8)),'region_color_luminance')
        self.assertFalse(np.array_equal(a,b))
        for profile in PROFILES:
            result=stylize(rgb,zero,profile)
            self.assertEqual(result.dtype,np.uint8);self.assertEqual(result.shape,rgb.shape)

    def test_invalid_mask_is_not_silently_used(self):
        rgb=np.zeros((8,8,3),np.uint8)
        for mask in (np.zeros((4,4)),np.full((8,8),np.nan),np.full((8,8),2)):
            with self.assertRaises(ValueError):stylize(rgb,mask,'identity')


if __name__=='__main__':unittest.main()
