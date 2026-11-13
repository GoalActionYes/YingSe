import importlib
import importlib.util
import unittest
import numpy as np
import torch


class SpatialRenderTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('spatial_render'))
        return importlib.import_module('spatial_render')

    def test_chunk_coordinates_use_full_image_rows(self):
        api=self.api();z=torch.full((1,2,4),.5)
        grid=api.coordinates(4,10,6,8,z)
        torch.testing.assert_close(grid[0,0,:,0,1],torch.tensor([.3,.5]))
        torch.testing.assert_close(grid[0,0,0,:,0],torch.tensor([-.75,-.25,.25,.75]))
        self.assertTrue(bool((grid[...,2]==0).all()))

    def test_zero_strength_preserves_bytes_without_engine_context(self):
        api=self.api();prepared=api.PreparedSpatial.__new__(api.PreparedSpatial)
        source=np.random.default_rng(81).integers(0,256,(43,61,3),dtype=np.uint8)
        output=prepared.apply(source,0)
        np.testing.assert_array_equal(output,source);self.assertIsNot(output,source)

    def test_invalid_strength_and_chunk_bounds_rejected(self):
        api=self.api();prepared=api.PreparedSpatial.__new__(api.PreparedSpatial);rgb=np.zeros((4,4,3),np.uint8)
        for strength in [-1,1.01,float('nan'),True]:
            with self.assertRaises(ValueError):prepared.apply(rgb,strength)
        with self.assertRaises(ValueError):api.coordinates(4,10,9,12,torch.ones(1,3,4))

    def test_candidate_is_explicit_and_default_checkpoint_stays_fixed(self):
        api=self.api();default,sha=api.checkpoint_for('app8');candidate,new_sha=api.checkpoint_for('multiref-v1')
        self.assertEqual(default.name,'epoch-8.pth');self.assertEqual(sha,api.CHECKPOINT_SHA)
        self.assertEqual(candidate.name,'epoch-6.pth');self.assertEqual(new_sha,'d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9')
        self.assertNotEqual(default,candidate)
        for invalid in ['unknown','../outside.pth',None]:
            with self.assertRaises(ValueError):api.checkpoint_for(invalid)


if __name__=='__main__':unittest.main()
