import unittest,importlib
import torch


class SemanticLocalTests(unittest.TestCase):
    def test_zero_output_and_gradient(self):
        a=importlib.import_module('semantic_local');h=a.SemanticCorrection();x=torch.rand(1,3,8,8);tokens=torch.rand(1,64,387)
        result=h(x,x,tokens,tokens);self.assertTrue(torch.equal(result,x));result.sum().backward();self.assertGreater(float(h.layers[-1].weight.grad.abs().sum()),0)

    def test_reference_permutation_invariance_and_pool_query_invariance(self):
        a=importlib.import_module('semantic_local');q=torch.rand(1,64,387);r=torch.rand(1,64,387);permutation=torch.randperm(64)
        torch.testing.assert_close(a.match(q,r),a.match(q,r[:,permutation]),atol=1e-6,rtol=1e-5)
        c=a.match(q,r,True);torch.testing.assert_close(c,c[:,:1].expand_as(c),atol=1e-6,rtol=1e-5)

    def test_local_matching_and_invalid_values(self):
        a=importlib.import_module('semantic_local');q=torch.zeros(1,64,387);r=q.clone();q[:,0,0]=1;q[:,1:,1]=1;r[:,:32,0]=1;r[:,32:,1]=1;r[:,:32,384]=1
        m=a.match(q,r);self.assertGreater(float(m[0,0,384]),float(m[0,1,384]))
        with self.assertRaises(ValueError):a.match(q,r*float('nan'))


if __name__=='__main__':unittest.main()
