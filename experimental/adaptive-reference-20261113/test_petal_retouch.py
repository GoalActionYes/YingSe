import unittest
import numpy as np


class RetouchTests(unittest.TestCase):
    def test_zero_transform_is_exact_and_brightness_has_expected_effect(self):
        from petal_retouch import retouch
        a=np.array([[[0,100,255],[40,80,120]]],np.uint8);before=a.copy()
        np.testing.assert_array_equal(retouch(a,np.zeros(5)),a)
        got=retouch(a,np.array([1.,0,0,0,0]))
        np.testing.assert_array_equal(got,np.array([[[18,118,255],[58,98,138]]],np.uint8))
        np.testing.assert_array_equal(a,before)

    def test_solve_zero_residual_degenerate_and_bounded(self):
        from petal_retouch import solve
        base=np.zeros(6);base[0]=1
        features=np.tile(base,(11,1))
        np.testing.assert_array_equal(solve(features,base),np.zeros(5))
        other=np.zeros(6);other[1]=1
        np.testing.assert_array_equal(solve(features,other),np.zeros(5))
        for i in range(5):
            features[1+2*i,i+1]=.2;features[2+2*i,i+1]=-.2
        u=solve(features,other)
        self.assertAlmostEqual(u[0],1.);np.testing.assert_array_equal(u[1:],np.zeros(4))
        u=solve(features,base)
        np.testing.assert_array_equal(u,np.zeros(5))

    def test_selection_obeys_cosine_gain_and_clipping_and_ties(self):
        from petal_retouch import select
        self.assertEqual(select([.2,.1,.15,.1],[.01,.02,.011,.01]),3)
        self.assertEqual(select([.2,.1995,.3,.4],[.01]*4),0)
        self.assertEqual(select([.2,.199,.3,.4],[.01]*4),0)
        self.assertEqual(select([.2,.1,.1,.3],[.01]*4),1)
        self.assertEqual(select([.2,.1,.15,.3],[.01,.015,.01,.01]),1)

    def test_invalid_inputs_fail_closed(self):
        from petal_retouch import retouch,solve,select
        a=np.zeros((2,3,3),np.uint8)
        for bad in [np.zeros(4),np.full(5,np.nan),np.ones(5)*1.01]:
            with self.assertRaises(ValueError):retouch(a,bad)
        with self.assertRaises(ValueError):retouch(a.astype(float),np.zeros(5))
        for bad in [np.zeros((11,6)),np.ones((10,6)),np.full((11,6),np.nan)]:
            with self.assertRaises(ValueError):solve(bad,np.ones(6))
        with self.assertRaises(ValueError):select([.2,.1],[.01]*4)
        with self.assertRaises(ValueError):select([.2,.1,.1,np.nan],[.01]*4)
