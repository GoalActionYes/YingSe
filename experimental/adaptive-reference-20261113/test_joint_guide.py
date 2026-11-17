import unittest
import numpy as np
from joint_guide import pooled_crossfit
from capacity_audit import ridge_fit


class JointGuideTests(unittest.TestCase):
    def test_equal_weight_solution_and_unfitted_pixels(self):
        rng=np.random.default_rng(71);a=rng.normal(size=(40,4));b=rng.normal(size=(40,4));coef=rng.normal(size=(4,3));ya=a@coef;yb=b@coef;m=np.arange(40)%2==0
        result=pooled_crossfit(a,ya,b,yb,m)
        expected=ridge_fit(np.r_[a,b],np.r_[ya,yb],np.r_[m,m])
        np.testing.assert_array_equal(result['fold_coefficients'][0],expected)
        np.testing.assert_allclose(result['a_crossfit'][~m],a[~m]@expected,atol=1e-14)
        changed_a=ya.copy();changed_b=yb.copy();changed_a[~m]+=50;changed_b[~m]-=90
        altered=pooled_crossfit(a,changed_a,b,changed_b,m)
        np.testing.assert_array_equal(altered['fold_coefficients'][0],expected)
        np.testing.assert_array_equal(result['a_crossfit'][~m],altered['a_crossfit'][~m])
        np.testing.assert_array_equal(result['b_crossfit'][~m],altered['b_crossfit'][~m])

    def test_conflicting_constant_maps_and_bad_shape(self):
        x=np.ones((6,1));m=np.array([1,0,1,0,1,0],bool)
        r=pooled_crossfit(x,np.zeros((6,3)),x,np.ones((6,3)),m)
        np.testing.assert_allclose(r['a_crossfit'],.5,atol=1e-5)
        with self.assertRaises(ValueError):pooled_crossfit(x,np.zeros((6,3)),x[:3],np.ones((3,3)),m)
