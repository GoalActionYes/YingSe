import unittest
import numpy as np
from reference_color_app.matrices import eigen2, solve_covariance, solve_matrices, negligible

def packets():
    s=np.zeros(63,np.float32); r=np.zeros(57,np.float32)
    s[:12]=r[:12]=np.tile(np.eye(2,dtype=np.float32).ravel(),3)
    s[12:39]=r[12:39]=np.tile(np.eye(3,dtype=np.float32).ravel(),3)
    s[39:51]=np.arange(12)+1
    s[51:54]=1; r[39:42]=1
    s[55:59]=[30,80,160,220]; s[60:63]=[20,100,200];r[51:53]=[0,255];r[54:57]=[30,110,210]
    return s,r

class MatrixTests(unittest.TestCase):
    def test_analytic_transport(self):
        a,e=solve_covariance(np.diag([4.,1.]),np.diag([16.,.25]))
        np.testing.assert_allclose(a[:2,:2],np.diag([2.,.5]),atol=5e-5)
        e,v=eigen2([[2,1],[1,2]])
        np.testing.assert_allclose(e,[3,1]); np.testing.assert_allclose(v.T@v,np.eye(2),atol=2e-7)
    def test_small_eigen_identity(self):
        for tiny in [0,1e-8]:
            a,e=solve_covariance(np.diag([1.,tiny]),np.eye(2))
            np.testing.assert_array_equal(a,np.eye(3))
    def test_degenerate_fresh_and_unknown_scratch(self):
        s,r=packets();s[:12]=0
        got=solve_matrices(s,r)
        np.testing.assert_array_equal(got['A'],np.tile(np.eye(3),(3,1,1)))
        np.testing.assert_array_equal(got['F'],got['R']@got['T'])
        self.assertTrue(np.isnan(got['eigenvalues']).all()); self.assertTrue(got['diagnostics'])
        r[:12]=0
        got=solve_matrices(s,r)
        np.testing.assert_array_equal(got['F'],np.zeros((3,3,3)));np.testing.assert_array_equal(got['flags'],[1,1,1])
    def test_direction_copy_ratio_and_outer_bounds(self):
        s,r=packets();s[14]=.2;r[14]=.3
        before=s.copy();got=solve_matrices(s,r)
        self.assertEqual(got['T'][0,0,2],np.float32(-.2))
        np.testing.assert_array_equal(got['C'].ravel(),s[39:51]);np.testing.assert_array_equal(s,before)
        np.testing.assert_allclose(got['F'],got['R']@(got['A']@got['T']),atol=1e-7)
        s[4:8]=[100,0,0,1];got=solve_matrices(s,r);np.testing.assert_array_equal(got['A'][1],np.eye(3))
        s,r=packets();r[:12]*=10000;got=solve_matrices(s,r)
        np.testing.assert_array_equal(got['A'][0],np.eye(3));self.assertAlmostEqual(float(got['max_excess']),.4)
    def test_eigenratio_threshold_and_outer_translation_branch(self):
        s,r=packets()
        # Ratios just below/above the literal threshold 70.
        for largest,reset in [(69.,False),(71.,True)]:
            s[4:8]=[largest,0,0,1];r[4:8]=[4*largest,0,0,4]
            got=solve_matrices(s,r)
            if reset: np.testing.assert_array_equal(got['A'][1],np.eye(3))
            else: self.assertGreater(float(got['A'][1,0,0]),1.9)
        s,r=packets();r[:4]*=16;s[14]=.2
        got=solve_matrices(s,r)
        self.assertLess(float(np.max(np.abs(got['A'][0]))),1.667)
        self.assertGreater(float(np.max(np.abs(got['A'][0]))),1.66)

    def test_threshold_and_invalid(self):
        self.assertTrue(negligible(np.zeros(2),np.zeros((2,2))))
        self.assertFalse(negligible(np.array([.0039,0],np.float32),np.zeros((2,2))))
        s,r=packets()
        for bad in [s[:-1],s.reshape(7,9),np.full(63,np.nan)]:
            with self.assertRaises(ValueError):solve_matrices(bad,r)
