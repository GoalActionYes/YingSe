import unittest
import numpy as np


class KernelTests(unittest.TestCase):
    def test_training_equation_and_constant_intercept(self):
        from condition_kernel import fit,predict
        x=np.arange(-2,3,dtype=float)[:,None];y=np.sin(x);m=fit(x,y)
        k=np.exp(-((x-x.T)**2)/(2*float(m['bandwidth_squared'])));kc=k-k.mean(0)[None]-k.mean(1)[:,None]+k.mean()
        np.testing.assert_allclose((kc+float(m['lambda_value'])*np.eye(5))@m['dual'],y-y.mean(0),atol=1e-12)
        constant=fit(x,np.full((5,2),3.));np.testing.assert_array_equal(predict(constant,np.array([[-3.],[.5]])),np.full((2,2),3.))

    def test_translation_rotation_and_label_offset(self):
        from condition_kernel import fit,predict
        x=np.array([[-2.,0],[-1,1],[0,-1],[1,2],[2,0]]);y=np.array([[-1.],[-.5],[0],[.5],[1.] ]);q=np.array([[-1.5,.2],[1.5,.2]]);m=fit(x,y);a=predict(m,q)
        rotation=np.array([[0.,-1],[1,0]]);other=fit(x@rotation+20,y+5)
        np.testing.assert_allclose(predict(other,q@rotation+20),a+5,atol=1e-12)
        self.assertLess(a[0,0],0);self.assertGreater(a[1,0],0)

    def test_invalid_shapes_and_degenerate_bandwidth(self):
        from condition_kernel import fit,predict
        for x,y in [(np.zeros((3,2)),np.ones((3,1))),(np.ones((1,2)),np.ones((1,1))),(np.ones((3,2)),np.ones((2,1))),(np.full((3,2),np.nan),np.ones((3,1)))]:
            with self.assertRaises(ValueError):fit(x,y)
        m=fit(np.arange(4.)[:,None],np.arange(4.)[:,None])
        with self.assertRaises(ValueError):predict(m,np.ones((2,2)))
