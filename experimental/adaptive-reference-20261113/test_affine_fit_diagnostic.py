import unittest,importlib
import numpy as np


class AffineFitTests(unittest.TestCase):
    def test_recover_known_bounded_affine_field(self):
        a=importlib.import_module('affine_fit_diagnostic');rng=np.random.default_rng(17)
        rgb=rng.integers(30,220,(16,16,3),dtype=np.uint8);p=rng.random((16,16));base=rgb.astype(float)/255
        x=a.design(rgb,p,True);coef=rng.normal(0,.15,(8,3));truth=base+.1*np.tanh((x@coef).reshape(base.shape))
        actual,_,_=a.fit(rgb,p,base,truth,True)
        self.assertLess(float(np.abs(actual-truth).mean()),1e-4)

    def test_three_branch_parameterization_equals_eight_basis(self):
        a=importlib.import_module('affine_fit_diagnostic');rng=np.random.default_rng(18);rgb=rng.integers(0,255,(8,8,3),dtype=np.uint8);p=rng.random((8,8));c=rng.normal(size=(3,4,3))
        x=a.design(rgb,p,False);full=x@c[0]+p.reshape(-1,1)*(x@c[1])+(1-p.reshape(-1,1))*(x@c[2])
        reduced=a.design(rgb,p,True)@np.concatenate([c[0]+c[2],c[1]-c[2]])
        np.testing.assert_allclose(full,reduced,atol=1e-12)

    def test_invalid_values_rejected_and_input_unchanged(self):
        a=importlib.import_module('affine_fit_diagnostic');rgb=np.full((8,8,3),128,np.uint8);p=np.zeros((8,8));before=rgb.copy();base=rgb/255
        a.fit(rgb,p,base,base,False);np.testing.assert_array_equal(rgb,before)
        with self.assertRaises(ValueError):a.fit(rgb,p+2,base,base,True)
        with self.assertRaises(ValueError):a.fit(rgb,p,base,base*np.nan,True)


if __name__=='__main__':unittest.main()
