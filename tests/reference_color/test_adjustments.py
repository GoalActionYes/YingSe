import unittest
import numpy as np
from reference_color_app.adjustments import normalize, transform, compose
from reference_color_app.contracts import Mapping
from reference_color_app.lut import apply_mapping


class AdjustmentTests(unittest.TestCase):
    def test_neutral_and_disabled(self):
        rgb=np.random.default_rng(8).random((20,20,3),dtype=np.float32)
        np.testing.assert_array_equal(transform(rgb,None),rgb)
        np.testing.assert_array_equal(transform(rgb,{'basic':{'enabled':False,'brightness':70}}),rgb)

    def test_invalid_states(self):
        for state in [True,[],{'version':2},{'unknown':0},{'basic':{'brightness':True}},
                      {'basic':{'brightness':float('nan')}},{'basic':{'brightness':101}},
                      {'hsl':{'red':[0,0]}},{'curves':{'r':[[0,0],[0,1]]}}]:
            with self.subTest(state=state),self.assertRaises(ValueError):normalize(state)

    def test_direction_and_bounds(self):
        rgb=np.full((4,4,3),.4,np.float32)
        self.assertGreater(transform(rgb,{'basic':{'brightness':40}}).mean(),rgb.mean())
        warm=transform(rgb,{'basic':{'temperature':40}})
        self.assertGreater(warm[0,0,0],warm[0,0,2])
        for name in ['temperature','tint','saturation','vibrance','brightness','contrast','highlights','shadows','whites','blacks','brilliance']:
            for value in [-100,100]:
                result=transform(rgb,{'basic':{name:value}})
                self.assertTrue(np.isfinite(result).all());self.assertTrue(((result>=0)&(result<=1)).all())

    def test_hsl_gray_and_red_wrap(self):
        gray=np.full((1,3,3),.5,np.float32)
        np.testing.assert_allclose(transform(gray,{'hsl':{'red':[100,100,0]}}),gray,atol=1e-6)
        red=np.array([[[.8,.02,.021],[.8,.021,.02]]],np.float32)
        result=transform(red,{'hsl':{'red':[20,15,10]}})
        np.testing.assert_allclose(result[0,0],result[0,1],atol=.005)

    def test_curves_channels_and_composition(self):
        rgb=np.full((1,2,3),.5,np.float32)
        adjusted=transform(rgb,{'curves':{'r':[[0,0],[.5,.75],[1,1]]}})
        self.assertAlmostEqual(float(adjusted[0,0,0]),.75,places=5)
        np.testing.assert_array_equal(adjusted[...,1:],rgb[...,1:])
        axis=np.linspace(0,1,64,dtype=np.float32)
        grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        mapping=Mapping(np.stack([grid,grid]),{})
        self.assertIs(compose(mapping,.8,None)[0],mapping)
        out,strength=compose(mapping,.8,{'basic':{'temperature':30,'brightness':15}})
        pixels=np.array([[[60,80,120],[180,90,40]]],np.uint8)
        expected=np.rint(transform(pixels.astype(np.float32)/255,{'basic':{'temperature':30,'brightness':15}})*255).astype(np.uint8)
        np.testing.assert_allclose(apply_mapping(pixels,out,strength),expected,atol=1)

    def test_selective_tones_and_color_range(self):
        gray=np.array([[[.1,.1,.1],[.9,.9,.9]]],np.float32)
        shadow=transform(gray,{'basic':{'shadows':40}})-gray
        highlight=gray-transform(gray,{'basic':{'highlights':-40}})
        self.assertGreater(shadow[0,0].mean(),shadow[0,1].mean())
        self.assertGreater(highlight[0,1].mean(),highlight[0,0].mean())
        rgb=np.array([[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]],np.float32)
        result=transform(rgb,{'hsl':{'green':[0,-70,0]}})
        np.testing.assert_allclose(result[:,[0,2]],rgb[:,[0,2]],atol=1e-6)
        self.assertLess(np.ptp(result[0,1]),np.ptp(rgb[0,1]))

    def test_vibrance_gray_and_disabled_groups(self):
        rgb=np.array([[[.5,.5,.5],[.6,.5,.5],[.95,.05,.05]]],np.float32)
        vib=transform(rgb,{'basic':{'vibrance':50}})
        sat=transform(rgb,{'basic':{'saturation':50}})
        np.testing.assert_allclose(vib[0,0],rgb[0,0],atol=1e-6)
        self.assertLess(np.linalg.norm(vib[0,2]-rgb[0,2]),np.linalg.norm(sat[0,2]-rgb[0,2]))
        state={'basic':{'enabled':False,'brightness':50},'hsl':{'enabled':False,'green':[100,100,100]},'curves':{'enabled':False,'r':[[0,1],[1,0]]}}
        np.testing.assert_array_equal(transform(rgb,state),rgb)

    def test_curve_validation_and_dense_finite_outputs(self):
        for points in [[[.1,0],[1,1]],[[0,0],[.5,2],[1,1]],[[0,0],[.6,.6],[.3,.3],[1,1]]]:
            with self.assertRaises(ValueError):normalize({'curves':{'rgb':points}})
        rgb=np.random.default_rng(9).random((32,32,3),dtype=np.float32)
        state={'basic':{key:100 if i%2 else -100 for i,key in enumerate(['temperature','tint','saturation','vibrance','brightness','contrast','highlights','shadows','whites','blacks','brilliance'])},'hsl':{key:[100,-100,100] for key in ['red','orange','yellow','green','cyan','blue','purple','magenta']},'curves':{'rgb':[[0,1],[.5,0],[1,1]]}}
        result=transform(rgb,state)
        self.assertTrue(np.isfinite(result).all());self.assertTrue(((result>=0)&(result<=1)).all())


if __name__=='__main__':unittest.main()
