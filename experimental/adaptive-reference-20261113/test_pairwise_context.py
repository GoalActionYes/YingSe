import unittest
import torch
from pairwise_context import pairwise_histograms,PairwiseContext
from joint_features import conditional_color


class PairwiseContextTests(unittest.TestCase):
    def test_pair_distributions_resolve_a_collision_in_marginals_and_conditional_means(self):
        a=torch.tensor([[[[.5,.5]], [[-.125,.125]], [[-.125,.125]]]])
        b=a.clone();b[:,2]=b[:,2].flip(-1)
        for channel in range(3):torch.testing.assert_close(a[:,channel].sort().values,b[:,channel].sort().values)
        torch.testing.assert_close(conditional_color(a),conditional_color(b))
        ha,hb=pairwise_histograms(a),pairwise_histograms(b)
        self.assertFalse(torch.equal(ha,hb));self.assertEqual(tuple(ha.shape),(1,3*16*16))
        torch.testing.assert_close(ha.square().reshape(1,3,-1).sum(-1),torch.ones(1,3))

    def test_joint_histograms_are_pixel_order_invariant_and_independent_per_batch(self):
        x=torch.tensor([[[[0.,1.]], [[-.5,.5]], [[.5,-.5]]]])
        h=pairwise_histograms(x);torch.testing.assert_close(h,pairwise_histograms(x.flip(-1)))
        batched=pairwise_histograms(torch.cat([x,torch.zeros_like(x)],0))
        torch.testing.assert_close(batched[0],h[0])
        torch.testing.assert_close(batched[1],pairwise_histograms(torch.zeros_like(x))[0])

    def test_adapter_starts_at_zero_and_gets_finite_gradients(self):
        layer=PairwiseContext();a=pairwise_histograms(torch.zeros(2,3,4,4));b=pairwise_histograms(torch.full((2,3,4,4),.25))
        output=layer(a,b);torch.testing.assert_close(output,torch.zeros(2,128));(output-1).square().mean().backward()
        self.assertTrue(torch.isfinite(layer.layers[-1].weight.grad).all());self.assertGreater(float(layer.layers[-1].weight.grad.abs().sum()),0)

    def test_invalid_inputs_are_rejected(self):
        for x in [torch.zeros(1,2,3,3),torch.full((1,3,3,3),float('nan')),torch.full((1,3,3,3),2.),torch.zeros(1,3,0,2),torch.zeros(1,3,3,3,dtype=torch.int64)]:
            with self.assertRaises(ValueError):pairwise_histograms(x)
        for bins in [True,1,33]:
            with self.assertRaises(ValueError):pairwise_histograms(torch.zeros(1,3,2,2),bins)


if __name__=='__main__':unittest.main()
