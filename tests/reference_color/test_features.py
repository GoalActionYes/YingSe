import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from reference_color_app.features import extract_features, region_masks, histogram

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'outputs/jianying-color/deep/test-native-cache'

class FeatureTests(unittest.TestCase):
    def test_audited_ordered_regions(self):
        cases = [
            ((0,150,140,130,120,255),125,(False,False,True)),
            ((0,150,140,130,120,255),135,(False,False,True)),
            ((0,150,140,130,120,255),145,(False,False,True)),
            ((100,20,40,60,80,255),30,(True,True,False)),
            ((0,20,40,60,100,70),80,(False,True,True)),
            ((0,150,140,130,120,200),201,(False,False,False)),
            ((0,150,140,130,120,125),125,(False,False,True)),
        ]
        for bounds,y,expected in cases:
            with self.subTest(bounds=bounds,y=y):
                self.assertEqual(tuple(bool(m[0]) for m in region_masks(np.array([y]),bounds)),expected)

    def test_builder_nonmonotonic_regions(self):
        q=np.zeros((1,200,3),np.uint8);q[:,:,0]=120
        bounds=histogram(q,1)[0]
        np.testing.assert_array_equal(bounds,[30,123,122,121,120,225])
        masks=np.array(region_masks(np.array([119,120,121,122,123,225,226]),bounds))
        np.testing.assert_array_equal(masks,[[1,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,1,1,1,1,1,0]])

    def test_sorted_regions_all_endpoints(self):
        y=np.arange(256)
        masks=region_masks(y,[10,30,60,100,170,225])
        for actual,expected in zip(masks,[(y>=10)&(y<60),(y>=30)&(y<170),(y>=100)&(y<=225)]):
            np.testing.assert_array_equal(actual,expected)

    def test_mode0_normal_layout_and_measured_native_mismatch(self):
        rgb=np.array(Image.open(ROOT/'outputs/jianying-color/test-kit/source.png').convert('RGB'))
        p=extract_features(rgb,0).values
        self.assertEqual(p.shape,(63,))
        self.assertEqual(p.dtype,np.dtype('<f4'))
        cov=p[:12].reshape(3,2,2)
        affine=p[12:39].reshape(3,3,3)
        inverse=p[39:51].reshape(3,2,2)
        # Independent mathematical/layout oracle, not probe-generated expected data.
        for matrix,inv in zip(cov,inverse):
            if np.linalg.det(matrix.astype(float))<=1e-5:
                np.testing.assert_array_equal(inv,np.eye(2))
            else:
                np.testing.assert_allclose(matrix@inv,np.eye(2),rtol=0,atol=2e-6)
        np.testing.assert_array_equal(affine[:,:,:2],np.broadcast_to(np.eye(3,dtype=np.float32)[:,:2],(3,3,2)))
        np.testing.assert_array_equal(affine[:,2,2],np.ones(3))
        native=np.fromfile(CACHE/'color_match_source_feature_1792064357214.match',dtype='<f4')
        # Source PNG is not encoded-video texture: these are measured mismatch limits.
        for lo,hi,limit in [(0,12,.00076),(12,39,.0029),(39,51,4.81),(51,54,5.78),(54,60,1),(60,63,.154)]:
            self.assertLessEqual(float(np.max(np.abs(p[lo:hi]-native[lo:hi]))),limit)

    def test_reference_truth(self):
        rgb = np.array(Image.open(CACHE / 'color_match_1792064357214.png').convert('RGB'))
        packet = extract_features(rgb, 1)
        self.assertEqual(packet.values.dtype, np.dtype('<f4'))
        self.assertEqual(packet.values.shape, (57,))
        np.testing.assert_array_equal(packet.values[42:48], [30,87,126,137,167,225])
        native = np.fromfile(CACHE / 'color_match_target_feature_1792064357214.match', dtype='<f4')
        # Fixed measured regression limits; independent native truth remains separate.
        for lo, hi, limit in [(0,12,.00018),(12,39,.00020),(39,42,.45),(42,48,0),(48,51,0),(51,53,0),(53,54,.001),(54,57,.20)]:
            self.assertLessEqual(float(np.max(np.abs(packet.values[lo:hi]-native[lo:hi]))), limit)

    def test_degenerate_inputs(self):
        for mode in (0,1):
            for rgb in (np.zeros((1,1,3),np.uint8), np.full((3,2,3),255,np.uint8), np.full((23,41,3),[255,0,0],np.uint8), np.full((21,22,3),[126,127,128],np.uint8)):
                p=extract_features(rgb,mode)
                self.assertEqual(p.values.shape,(63 if mode==0 else 57,))
                self.assertTrue(np.isfinite(p.values).all())

    def test_invalid_inputs(self):
        for rgb in (np.empty((0,2,3),np.uint8),np.zeros((2,2),np.uint8),np.zeros((2,2,4),np.uint8),np.zeros((2,2,3),np.float32),np.zeros((1,10000,3),np.uint8)):
            with self.assertRaises((ValueError,TypeError)): extract_features(rgb,0)
        for mode in (True,False,-1,2,1.0,'1',None):
            with self.assertRaises((ValueError,TypeError)): extract_features(np.zeros((1,1,3),np.uint8),mode)

    def test_mode_isolation(self):
        rgb=np.full((21,31,3),[200,60,10],np.uint8)
        a=extract_features(rgb,0).values.copy()
        extract_features(rgb,1)
        np.testing.assert_array_equal(a,extract_features(rgb,0).values)

if __name__=='__main__': unittest.main()
