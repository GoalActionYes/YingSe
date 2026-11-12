import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional model environment required')
class PreviewChecks(unittest.TestCase):
    def test_matches_stable_two_layer_response(self):
        import torch
        from gpu_preview import PreviewRenderer
        from reference_color_app.contracts import Mapping
        from reference_color_app.lut import apply_mapping
        axis=np.linspace(0,1,64,dtype=np.float32)
        rgb=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        first=np.clip(rgb*[.8,1,.9]+[.05,0,.03],0,1).astype(np.float32)
        second=np.clip(rgb[...,::-1]*[.9,1,.8]+[0,.02,.04],0,1).astype(np.float32)
        mapping=Mapping(np.stack([first,second]),{})
        frame=np.random.default_rng(2).integers(0,256,(32,48,3),dtype=np.uint8)
        renderer=PreviewRenderer('cuda' if torch.cuda.is_available() else 'cpu')
        for strength in (0,.5,.8,.9,1):
            expected=apply_mapping(frame,mapping,strength)
            actual=renderer.apply(frame,mapping,strength)
            self.assertLessEqual(np.abs(actual.astype(int)-expected).max(),1)
        self.assertTrue(np.array_equal(renderer.apply(frame,mapping,0),frame))


if __name__=='__main__':unittest.main()
