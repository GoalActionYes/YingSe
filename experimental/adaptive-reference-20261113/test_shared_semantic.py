import unittest
import numpy as np
from shared_semantic_crops import SharedCanonicalCandidate


class NoInference:
    def predict(self,*args):raise AssertionError('Identity should not call model')


class SharedChecks(unittest.TestCase):
    def test_identity_bypasses_all_model_inference(self):
        rgb=np.full((32,48,3),128,np.uint8);mask=np.zeros((32,48))
        result,global_rgb,diagnostic=SharedCanonicalCandidate(NoInference()).predict(rgb,rgb,mask,mask)
        np.testing.assert_array_equal(result,rgb);np.testing.assert_array_equal(global_rgb,rgb)
        self.assertTrue(diagnostic['exact_identity'])


if __name__=='__main__':unittest.main()
