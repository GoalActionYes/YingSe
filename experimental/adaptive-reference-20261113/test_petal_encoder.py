import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F


class EncoderTests(unittest.TestCase):
    def model(self, output=None):
        captured=[]
        def style(x):
            captured.append(x.clone())
            return {'z': torch.tensor([[3.,4.],[4.,3.]]) if output is None else output}
        return SimpleNamespace(device='cpu',api=SimpleNamespace(rgb2lab=lambda x:x*2+.25),style=style),captured

    def test_official_preprocessing_and_normalization(self):
        from petal_encoder import encode
        a=np.array([[[0,127,255],[50,80,100]]],np.uint8)
        b=np.full((3,2,3),128,np.uint8);before=a.copy();model,captured=self.model()
        got=encode(model,[a,b])
        expected=torch.cat([F.interpolate(torch.from_numpy(v.astype(np.float32)/255).permute(2,0,1)[None],size=(512,512),mode='bilinear',align_corners=False)*2+.25 for v in [a,b]])
        torch.testing.assert_close(captured[0],expected,rtol=0,atol=0)
        torch.testing.assert_close(got,torch.tensor([[.6,.8],[.8,.6]]))
        np.testing.assert_array_equal(a,before)

    def test_invalid_outputs_rejected(self):
        from petal_encoder import encode
        rgb=np.zeros((2,3,3),np.uint8)
        for value in [torch.zeros(2,2),torch.full((2,2),float('nan')),torch.ones(1,2),torch.ones(2,2,1)]:
            model,_=self.model(value)
            with self.assertRaises(ValueError):encode(model,[rgb,rgb])

    def test_invalid_input_rejected(self):
        from petal_encoder import encode
        model,_=self.model()
        for inputs in [[],[np.zeros((2,3,3),float)]]:
            with self.assertRaises(ValueError):encode(model,inputs)
