import unittest
import torch
from tone_relations import source_tone_weights,tone_descriptor,tone_relation_losses


class ToneRelationTests(unittest.TestCase):
    def test_soft_regions_are_fixed_by_source_and_sum_to_one(self):
        source=torch.rand(4,3,9,11,requires_grad=True);weights=source_tone_weights(source)
        self.assertEqual(tuple(weights.shape),(4,4,9,11));self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights[:,1:].sum(1),torch.ones(4,9,11));self.assertTrue(bool((weights>=0).all()))

    def test_exact_prediction_and_gray_missing_regions_are_finite(self):
        source=torch.full((4,3,8,8),.5);truth=torch.full_like(source,.3);prediction=truth.clone().requires_grad_()
        absolute,paired=tone_relation_losses(prediction,truth,source)
        self.assertEqual(float(absolute),0);self.assertEqual(float(paired),0)
        (absolute+paired).backward();self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        features=tone_descriptor(truth,source_tone_weights(source));torch.testing.assert_close(features[:,:,0],torch.full((4,4),.3));self.assertTrue(bool(torch.isfinite(features).all()))

    def test_wrong_reference_response_has_positive_loss_and_nonzero_finite_gradient(self):
        source=torch.rand(2,3,12,12).repeat_interleave(2,0);truth=source.clone();truth[1]=torch.clamp(truth[1]+.2,0,1);truth[3,:,6:]=torch.clamp(truth[3,:,6:]-.15,0,1)
        prediction=source.clone().requires_grad_();absolute,paired=tone_relation_losses(prediction,truth,source)
        self.assertGreater(float(absolute),0);self.assertGreater(float(paired),0)
        (absolute*.25+paired*.5).backward();self.assertTrue(bool(torch.isfinite(prediction.grad).all()));self.assertGreater(float(prediction.grad.abs().sum()),0)

    def test_opposite_shadow_highlight_colors_are_detected_despite_equal_global_rgb_means(self):
        source=torch.full((4,3,8,8),.25,dtype=torch.float64);source[:,:,4:]=.75
        tint=source.new_tensor([.06,-.06*(.2126-.0722)/.7152,-.06])[None,:,None,None]
        sign=torch.ones(1,1,8,8,dtype=source.dtype);sign[:,:,4:]=-1
        truth=source.clone();truth[::2]+=sign*tint;truth[1::2]-=sign*tint
        torch.testing.assert_close(truth.mean((-2,-1)),source.mean((-2,-1)))
        features=tone_descriptor(truth,source_tone_weights(source));torch.testing.assert_close(features[::2,0],features[1::2,0])
        absolute,paired=tone_relation_losses(source,truth,source);self.assertGreater(float(absolute),.001);self.assertGreater(float(paired),.001)

    def test_joint_spatial_permutation_and_pair_reversal_preserve_loss(self):
        source=torch.rand(2,3,8,8).repeat_interleave(2,0);truth=torch.rand_like(source);prediction=torch.rand_like(source)
        original=tone_relation_losses(prediction,truth,source);order=torch.randperm(64)
        permute=lambda x:x.flatten(2)[:,:,order].reshape_as(x)
        for a,b in zip(original,tone_relation_losses(permute(prediction),permute(truth),permute(source))):torch.testing.assert_close(a,b)
        reverse=[1,0,3,2]
        for a,b in zip(original,tone_relation_losses(prediction[reverse],truth[reverse],source[reverse])):torch.testing.assert_close(a,b)

    def test_unpaired_content_and_invalid_rgb_are_rejected(self):
        source=torch.rand(4,3,8,8)
        with self.assertRaises(ValueError):tone_relation_losses(source,source,source)
        for value in [torch.zeros(4,2,8,8),torch.full((4,3,8,8),float('nan')),torch.full((4,3,8,8),1.1)]:
            with self.assertRaises(ValueError):source_tone_weights(value)

    def test_subcode_roundoff_does_not_create_a_full_magnitude_l1_gradient(self):
        source=torch.full((4,3,8,8),.5);truth=torch.full_like(source,.4);prediction=(truth+1e-7).requires_grad_()
        absolute,paired=tone_relation_losses(prediction,truth,source);(absolute*.25+paired*.5).backward()
        self.assertLess(float(prediction.grad.abs().sum()),.001)


if __name__=='__main__':unittest.main()
