import importlib
import unittest
import torch


class InternalAffineTests(unittest.TestCase):
    def api(self):return importlib.import_module('internal_affine_stage')
    def model(self):
        class Model:
            def pre_guide(self,x):return x
        return Model()
    def test_known_diagonal_bias_and_alpha_normalization(self):
        x=torch.tensor([.2,.5,.8]).reshape(1,3,1,1).expand(2,3,2,3);c=torch.zeros((2,13,2,3));c[:,12]=3
        for i in [0,4,8]:c[:,i]=6
        c[:,9]=-.3;c[:,10]=.6;c[:,11]=.9
        actual=self.api().internal_affine(self.model(),c,x)
        expected=torch.tensor([-1.3,.2,1.5]).reshape(1,3,1,1).expand_as(x)
        torch.testing.assert_close(actual,expected)
    def test_batch_and_spatial_bias_do_not_mix_and_gradients_finite(self):
        x=torch.full((2,3,2,2),.5);c=torch.zeros((2,13,2,2));c[:,12]=1;c[0,9]=torch.tensor([[0.,1.],[2.,3.]]);c[1,10]=-2;c.requires_grad_()
        out=self.api().internal_affine(self.model(),c,x);self.assertEqual(out[1,1,0,0].item(),-2);self.assertEqual(out[0,0,1,1].item(),3);self.assertEqual(out[0,1,1,1].item(),0)
        out.sum().backward();self.assertTrue(torch.isfinite(c.grad).all());self.assertGreater(c.grad.abs().sum().item(),0)
    def test_invalid_shape_alpha_and_nonfinite_stop(self):
        api=self.api();x=torch.zeros((1,3,2,2));c=torch.zeros((1,13,2,2));c[:,12]=1
        for bad in [c[:,:12],c*0,c*float('nan')]:
            with self.assertRaises(ValueError):api.internal_affine(self.model(),bad,x)


if __name__=='__main__':unittest.main()
