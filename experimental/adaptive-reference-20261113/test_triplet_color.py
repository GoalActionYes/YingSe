import importlib.util
import importlib
import unittest
import numpy as np
from PIL import Image,ImageCms


class ColorPreparationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('triplet_color'))
        return importlib.import_module('triplet_color')

    def test_srgb_profile_preserves_pixels_and_source(self):
        api=self.api();rgb=np.random.default_rng(18).integers(0,256,(16,24,3),dtype=np.uint8)
        image=Image.fromarray(rgb);image.info['icc_profile']=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        result,info=api.to_srgb(image)
        np.testing.assert_array_equal(np.array(result),rgb);np.testing.assert_array_equal(np.array(image),rgb)
        self.assertEqual(info['policy'],'embedded ICC to sRGB')

    def test_missing_profile_is_explicit_assumption(self):
        result,info=self.api().to_srgb(Image.new('RGB',(12,12),(18,31,190)))
        self.assertEqual(result.getpixel((0,0)),(18,31,190));self.assertEqual(info['policy'],'untagged assumed sRGB')

    def test_corrupt_profile_and_unsupported_modes_rejected(self):
        api=self.api();image=Image.new('RGB',(12,12));image.info['icc_profile']=b'broken'
        with self.assertRaises(ValueError):api.to_srgb(image)
        with self.assertRaises(ValueError):api.to_srgb(Image.new('I',(12,12)))


if __name__=='__main__':unittest.main()
