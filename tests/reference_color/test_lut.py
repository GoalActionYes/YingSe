import unittest
import numpy as np
from reference_color_app.contracts import Mapping, FeaturePacket
from reference_color_app.lut import apply_mapping, cube_text, build_mapping, _identity_cube, _sample

class LUTTests(unittest.TestCase):
    def identity(self):
        grid=np.stack(np.meshgrid(*[np.linspace(0,1,64,dtype=np.float32)]*3,indexing='ij'),axis=-1)
        return Mapping(np.stack([grid,grid]),{})
    def test_identity_corners_and_slice_boundaries(self):
        m=self.identity()
        rgb=np.array([[r,g,b] for r in (0,255) for g in (0,255) for b in range(256)],np.uint8)
        for t in (0,.5,.8,1):np.testing.assert_array_equal(apply_mapping(rgb,m,t),rgb)
        grid=_identity_cube()
        for b in range(64):
            np.testing.assert_array_equal(grid[0,63,b],np.array([0,255,int(np.float32(b)/np.float32(63)*np.float32(255))],np.float32)/np.float32(255))
    def test_final_strength_truth(self):
        layers=np.empty((2,64,64,64,3),np.float32);layers[0]=[.2,.4,.6];layers[1]=[.8,.6,.4]
        m=Mapping(layers,{})
        x=np.array([[0,100,255]],np.uint8)
        np.testing.assert_array_equal(apply_mapping(x,m,0),x)
        np.testing.assert_array_equal(apply_mapping(x,m,.4),np.rint((x/255*.5+np.array([.2,.4,.6])*.5)*255).astype(np.uint8))
        np.testing.assert_allclose(apply_mapping(x,m,.9),[[128,128,128]],atol=1)
        np.testing.assert_array_equal(apply_mapping(x,m,1),[[204,153,102]])
    def test_cube_order_and_unquantized(self):
        lines=cube_text(self.identity(),.5).splitlines();data=np.array([[float(x) for x in line.split()] for line in lines if line and line[0].isdigit()])
        np.testing.assert_allclose(data[1],[1/63,0,0],atol=1e-8)
        np.testing.assert_allclose(data[64],[0,1/63,0],atol=1e-8)
        np.testing.assert_allclose(data[4096],[0,0,1/63],atol=1e-8)
    def test_chunk_boundary_matches_same_unblocked_math(self):
        from reference_color_app.lut import _compose
        rgb=np.random.default_rng(7).integers(0,256,(257,256,3),dtype=np.uint8)
        m=self.identity()
        m.layers[0,...,0] *= np.float32(.7)
        expected=np.rint(np.clip(_compose(m.layers,rgb.astype(np.float32)/np.float32(255),.9),0,1)*np.float32(255)).astype(np.uint8)
        np.testing.assert_array_equal(apply_mapping(rgb,m,.9),expected)

    def test_unsorted_threshold_glsl_first_branch_wins(self):
        from reference_color_app.lut import _luma_weights
        bounds=np.array([80,130,129,179],np.float32)/np.float32(255)
        y=np.array([129.5],np.float32)/np.float32(255)
        w=_luma_weights(y,bounds)
        np.testing.assert_allclose(w,[[.01,.99,0]],atol=2e-7)

    def test_invalid(self):
        m=self.identity();x=np.zeros((1,1,3),np.uint8)
        for t in (-1,2,float('nan'),float('inf')):
            with self.assertRaises(ValueError):apply_mapping(x,m,t)
        with self.assertRaises(ValueError):apply_mapping(x.astype(float),m,1)
        with self.assertRaises(ValueError):apply_mapping(x,Mapping(m.layers[:1],{}),0)
        with self.assertRaises(ValueError):cube_text(m,1,32)
        with self.assertRaises(ValueError):build_mapping(FeaturePacket(1,np.zeros(63),{}),FeaturePacket(1,np.zeros(57),{}))

class FrozenGPUChecks(unittest.TestCase):
    def test_frozen_system_gl_color_and_gray(self):
        from pathlib import Path
        from PIL import Image
        from reference_color_app.features import extract_features
        root=Path(__file__).resolve().parents[2]
        cache=root/'outputs/jianying-color/deep/test-native-cache'
        pairs=[(FeaturePacket(0,np.frombuffer(next(cache.glob('*source_feature*.match')).read_bytes(),'<f4'),{}),FeaturePacket(1,np.frombuffer(next(cache.glob('*target_feature*.match')).read_bytes(),'<f4'),{}),'system-gl-probe')]
        gray=np.full((40,40,3),128,np.uint8)
        pairs.append((extract_features(gray,0),extract_features(gray,1),'system-gl-probe-without-uv'))
        for s,r,directory in pairs:
            mapping=build_mapping(s,r)
            for i,t in enumerate((.8,1)):
                atlas=np.asarray(Image.open(root/f'outputs/jianying-color/deep/{directory}/gl-layer-{t}.png').convert('RGB'))
                expected=np.stack([atlas[(b//8)*64:(b//8+1)*64,(b%8)*64:(b%8+1)*64].transpose(1,0,2) for b in range(64)],axis=2)
                actual=np.rint(mapping.layers[i]*255).astype(np.uint8)
                self.assertLessEqual(np.abs(actual.astype(int)-expected.astype(int)).max(),1,(directory,t))
