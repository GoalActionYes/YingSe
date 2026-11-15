import unittest
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace


class PetalModelTests(unittest.TestCase):
    def test_author_half_up_quantization_and_clipping(self):
        from petal_model import rgb8
        value=torch.tensor([-.1,.5/255,254.5/255,1.2],dtype=torch.float32).reshape(1,1,1,4).expand(1,3,1,4)
        out=rgb8(value)
        np.testing.assert_array_equal(out[0,:,0],np.array([0,1,255,255],np.uint8))

    def test_nonfinite_wrongshape_and_dtype_fail(self):
        from petal_model import rgb8,tensor
        for value in [torch.zeros(1,2,3,4),torch.zeros(1,3,0,4),torch.full((1,3,2,3),float('nan'))]:
            with self.assertRaises(ValueError):rgb8(value)
        with self.assertRaises(ValueError):tensor(np.zeros((2,3,3),float),'cpu')

    def test_source_tensor_range_shape_and_no_mutation(self):
        from petal_model import tensor
        value=np.array([[[0,127,255],[12,80,143]]],np.uint8);saved=value.copy()
        t=tensor(value,'cpu')
        self.assertEqual(t.dtype,torch.float32);self.assertEqual(tuple(t.shape),(1,3,1,2))
        np.testing.assert_allclose(t[0,:,0,0].numpy(),[0,127/255,1],atol=1e-7)
        np.testing.assert_array_equal(value,saved)

    def test_dependency_origin_and_version_fail_closed(self):
        from petal_model import verify_dependency
        wheel=Path('isolated/kornia.whl').resolve()
        correct=SimpleNamespace(__file__=str(wheel/'kornia/__init__.py'),__version__='0.7.0')
        verify_dependency(correct,wheel)
        for invalid in [SimpleNamespace(__file__='other/kornia/__init__.py',__version__='0.7.0'),SimpleNamespace(__file__=correct.__file__,__version__='0.8.0')]:
            with self.assertRaises(ValueError):verify_dependency(invalid,wheel)
