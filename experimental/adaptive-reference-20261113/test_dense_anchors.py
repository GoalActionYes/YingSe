import importlib
import importlib.util
import unittest
import torch


class DenseAnchorTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('dense_anchors'))
        return importlib.import_module('dense_anchors')

    def test_distinct_features_route_reference_colors(self):
        api=self.api()
        f=torch.eye(4)
        colors=torch.tensor([[.1,.2,.3],[.7,.2,.1],[.2,.7,.1],[.1,.2,.8]])
        anchors,confidence=api.match_anchors(f,f,colors)
        torch.testing.assert_close(anchors,colors,atol=1e-5,rtol=1e-5)
        self.assertTrue(bool((confidence>.99).all()))

    def test_no_match_keeps_base_and_zero_strength_is_exact(self):
        api=self.api()
        torch.manual_seed(2)
        image=torch.rand(1,3,28,28)
        anchors=torch.zeros(1,3,2,2)
        confidence=torch.zeros(1,1,2,2)
        out,_=api.color_from_anchors(image,anchors,confidence,spatial=True)
        self.assertTrue(torch.equal(out,image))
        out,_=api.color_from_anchors(image,anchors,torch.ones_like(confidence),strength=0,spatial=True)
        self.assertTrue(torch.equal(out,image))

    def test_same_rgb_can_have_different_regional_results(self):
        api=self.api()
        image=torch.full((1,3,56,56),.5)
        anchors=torch.full((1,3,4,4),.5)
        anchors[:,0,:,:2]=.65; anchors[:,2,:,2:]=.65
        confidence=torch.ones(1,1,4,4)
        global_rgb,_=api.color_from_anchors(image,anchors,confidence,spatial=False)
        local_rgb,_=api.color_from_anchors(image,anchors,confidence,spatial=True)
        self.assertLess(float((global_rgb[:,:,:,0]-global_rgb[:,:,:,-1]).abs().max()),1e-6)
        self.assertGreater(float(local_rgb[0,0,28,0]),float(local_rgb[0,0,28,-1])+.005)
        self.assertGreater(float(local_rgb[0,2,28,-1]),float(local_rgb[0,2,28,0])+.005)

    def test_guided_constant_and_output_are_finite_and_bounded(self):
        api=self.api()
        guide=torch.rand(1,3,63,87)
        field=torch.full((1,3,5,7),.1)
        up=api.guided_upsample(guide,field)
        self.assertEqual(up.shape,guide.shape)
        torch.testing.assert_close(up,torch.full_like(up,.1),atol=2e-5,rtol=2e-5)
        output,_=api.color_from_anchors(guide,torch.rand(1,3,5,7),torch.ones(1,1,5,7),spatial=True)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(bool(((output>=0)&(output<=1)).all()))

    def test_invalid_match_and_control_data_are_rejected(self):
        api=self.api()
        with self.assertRaises(ValueError): api.match_anchors(torch.full((3,4),float('nan')),torch.eye(4),torch.rand(4,3))
        with self.assertRaises(ValueError): api.color_from_anchors(torch.rand(1,3,28,28),torch.rand(1,3,2,2),torch.full((1,1,2,2),1.1))

    def test_convex_white_anchor_roundoff_stays_in_unit_range(self):
        api=self.api(); torch.manual_seed(171)
        f=torch.randn(193,8)
        anchors,_=api.match_anchors(f,f,torch.ones(193,3))
        self.assertTrue(bool(((anchors>=0)&(anchors<=1)).all()))

    def test_guidance_channel_permutation_does_not_change_field(self):
        api=self.api(); torch.manual_seed(96)
        guide=torch.rand(1,3,48,64)
        field=torch.rand(1,3,6,8)*.2-.1
        a=api.guided_upsample(guide,field)
        b=api.guided_upsample(guide[:,[2,0,1]],field)
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
