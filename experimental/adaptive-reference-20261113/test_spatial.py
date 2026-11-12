import unittest
import numpy as np
from spatial import fit, apply


class SpatialChecks(unittest.TestCase):
    def setUp(self):
        self.source = np.random.default_rng(42).integers(40, 210, (96,128,3), dtype=np.uint8)

    def test_smooth_regional_color_luminance_and_saturation(self):
        x=self.source.astype(float)
        w=np.linspace(0,1,128)[None,:,None]
        gray=(x @ [.299,.587,.114])[...,None]
        desired=np.rint(np.clip(gray+(x-gray)*(.6+.5*w)+20*(2*w-1)+
                               (2*w-1)*np.array([12,0,-12]),0,255)).astype(np.uint8)
        model=fit(self.source,desired)
        self.assertTrue(model['accepted'],model)
        self.assertLess(np.abs(apply(self.source,model).astype(float)-desired).mean(),1)

    def test_identity_and_uniform_transform_defer_to_global(self):
        self.assertFalse(fit(self.source,self.source)['accepted'])
        self.assertFalse(fit(self.source,self.source+10)['accepted'])

    def test_different_content_and_flat_rejected(self):
        other=np.random.default_rng(88).integers(40,210,self.source.shape,dtype=np.uint8)
        self.assertFalse(fit(self.source,other)['accepted'])
        flat=np.full_like(self.source,128)
        self.assertFalse(fit(flat,flat+10)['accepted'])

    def test_local_replacement_rejected(self):
        changed=self.source.copy()
        changed[24:56,24:56]=np.random.default_rng(4).integers(40,210,(32,32,3),dtype=np.uint8)
        self.assertFalse(fit(self.source,changed)['accepted'])

    def test_zero_strength_and_invalid_inputs(self):
        reference=self.source.copy().astype(float)
        reference+=np.linspace(-20,20,128)[None,:,None]
        model=fit(self.source,np.rint(reference).astype(np.uint8))
        self.assertTrue(model['accepted'])
        self.assertTrue(np.array_equal(apply(self.source,model,0),self.source))
        with self.assertRaises(ValueError): apply(self.source,model,float('nan'))
        with self.assertRaises(ValueError): fit(self.source.astype(float),self.source)

    def test_application_matches_independent_bilinear_node_interpolation(self):
        from spatial import VERSION
        nodes=np.zeros((3,3,4,3))
        for j in range(3):
            for i in range(3):
                nodes[j,i,:3]=np.eye(3)
                nodes[j,i,3]=np.array([j-1,i-1,j+i-2])*.03
        model=dict(accepted=True,version=VERSION,coefficients=nodes.reshape(36,3).tolist())
        result=apply(self.source,model)
        for row,col in [(0,0),(31,73),(95,127)]:
            y=row/95*2;x=col/127*2
            offset=np.array([y-1,x-1,y+x-2])*.03*255
            expected=np.rint(np.clip(self.source[row,col]+offset,0,255)).astype(np.uint8)
            np.testing.assert_array_equal(result[row,col],expected)


if __name__=='__main__': unittest.main()
