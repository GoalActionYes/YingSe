import unittest
import numpy as np
from tone_response import tone_signature,tone_response


class ToneResponseTests(unittest.TestCase):
    def test_identity_and_correct_change_have_zero_error(self):
        a=np.full((16,16,3),60,np.uint8);b=np.full_like(a,140)
        self.assertTrue(all(v==0 for v in tone_response(a,b,a,b)['absolute_delta_error'].values()))
        self.assertAlmostEqual(tone_signature(a)['brightness'],60,places=4)

    def test_no_change_does_not_fake_correct_response(self):
        a=np.full((16,16,3),60,np.uint8);b=np.full_like(a,140)
        result=tone_response(a,a,a,b)
        self.assertAlmostEqual(result['absolute_delta_error']['brightness'],80,places=4)
        self.assertFalse(result['direction_correct']['brightness'])
        self.assertIsNone(result['direction_correct']['saturation'])

    def test_contrast_and_chroma_are_separate_and_invalid_input_rejected(self):
        image=np.full((16,16,3),128,np.uint8);image[:8]=[40,80,140];image[8:]=[200,160,100]
        result=tone_signature(image)
        self.assertGreater(result['contrast'],10);self.assertGreater(result['saturation'],10)
        for bad in [image.astype(float),image[:,:,:2],np.empty((0,2,3),np.uint8)]:
            with self.assertRaises(ValueError):tone_signature(bad)


if __name__=='__main__':unittest.main()
