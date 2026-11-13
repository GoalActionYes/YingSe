"""Information separation and recovery tests for reference-original diagnosis."""
import unittest
import numpy as np
from reference_transfer_probe import crossfit_transfer

class ReferenceTransferTests(unittest.TestCase):
    def test_known_transform_transfers_to_different_content_and_colors(self):
        rng=np.random.default_rng(20261121)
        reference=np.column_stack([rng.uniform(.1,.6,(96,3)),np.ones(96)])
        target=np.column_stack([rng.uniform(.4,.9,(47,3)),np.ones(47)])
        matrix=np.array([[.8,.1,0],[0,.9,.1],[.1,0,.7],[.03,-.02,.05]])
        mask=np.arange(96)%2==0
        result=crossfit_transfer(reference,reference@matrix,target,mask)
        np.testing.assert_allclose(result['reference_crossfit'],reference@matrix,atol=.001)
        np.testing.assert_allclose(result['target_prediction'],target@matrix,atol=.002)

    def test_target_does_not_influence_reference_fitted_coefficients(self):
        rng=np.random.default_rng(11);x=rng.random((80,4));truth=rng.random((80,3));mask=np.arange(80)%2==0
        a=crossfit_transfer(x,truth,rng.random((33,4)),mask)
        b=crossfit_transfer(x,truth,rng.random((17,4))*10,mask)
        np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
        np.testing.assert_array_equal(a['reference_crossfit'],b['reference_crossfit'])

    def test_fold_prediction_excludes_its_reference_truth(self):
        rng=np.random.default_rng(3);x=rng.random((80,4));truth=rng.random((80,3));mask=np.arange(80)%2==0;changed=truth.copy();changed[~mask]=100
        a=crossfit_transfer(x,truth,x,mask);b=crossfit_transfer(x,changed,x,mask)
        np.testing.assert_array_equal(a['reference_crossfit'][~mask],b['reference_crossfit'][~mask])

    def test_fixed_transform_is_position_independent(self):
        rng=np.random.default_rng(5);x=rng.random((80,4));truth=rng.random((80,3));target=rng.random((40,4));order=rng.permutation(40);mask=np.arange(80)%2==0
        a=crossfit_transfer(x,truth,target,mask);b=crossfit_transfer(x,truth,target[order],mask)
        np.testing.assert_array_equal(a['target_prediction'][order],b['target_prediction'])

    def test_invalid_target_and_reference_features_fail(self):
        x=np.ones((8,4));truth=np.ones((8,3));mask=np.arange(8)%2==0
        for target in [np.ones((8,3)),np.ones((0,4)),np.full((8,4),np.nan)]:
            with self.assertRaises(ValueError):crossfit_transfer(x,truth,target,mask)
        with self.assertRaises(ValueError):crossfit_transfer(x,truth,x,np.ones(8,bool))

if __name__=='__main__':unittest.main()
