import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional model environment required')
class GPUMapChecks(unittest.TestCase):
    def test_identity_and_zero_strength(self):
        import torch
        from gpu_mapping import PreparedMapping
        coords=torch.linspace(0,1,17)
        z,y,x=torch.meshgrid(coords,coords,coords,indexing='ij')
        identity=torch.stack([x,y,z])[None]
        rgb=np.random.default_rng(3).integers(0,256,(32,48,3),dtype=np.uint8)
        mapping=PreparedMapping(identity,identity)
        result=mapping.apply(rgb)
        self.assertLessEqual(np.abs(result.astype(int)-rgb).max(),1)
        altered=PreparedMapping(identity,(identity*.8+.1))
        self.assertTrue(np.array_equal(altered.apply(rgb,0),rgb))
        with self.assertRaises(ValueError):altered.apply(rgb,float('nan'))

    def test_two_stage_order_and_channels(self):
        import torch
        from gpu_mapping import PreparedMapping
        axis=torch.linspace(0,1,17)
        z,y,x=torch.meshgrid(axis,axis,axis,indexing='ij')
        identity=torch.stack([x,y,z])[None]
        first=identity.clone();first[:,0]*=.5
        second=identity.clone();second[:,0]+=.1
        rgb=np.full((16,16,3),[200,100,50],np.uint8)
        output=PreparedMapping(first,second).apply(rgb)
        np.testing.assert_allclose(output[0,0],[126,100,50],atol=1)


if __name__=='__main__':unittest.main()
