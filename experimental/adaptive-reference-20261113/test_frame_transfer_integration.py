import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from spatial_render import PreparedSpatial


class FrameTransferIntegrationTests(unittest.TestCase):
    def test_actual_full_strength_uses_frame_renderer_and_256_default(self):
        rgb=np.zeros((3,4,3),np.uint8);prepared=PreparedSpatial(SimpleNamespace(),torch.ones(1,13,8,16,16),{})
        with patch('frame_transfer.render',return_value=rgb) as render:
            np.testing.assert_array_equal(prepared.apply(rgb,1),rgb)
            render.assert_called_once_with(prepared,rgb,256)

    def test_helper_source_changes_engine_identity(self):
        from spatial_store import SpatialStore
        from reference_color_app import pipeline
        original=pipeline.file_sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);engine=SimpleNamespace(checkpoint_sha='test')
            with patch.object(pipeline,'file_sha256',side_effect=lambda p:'version-A' if p.name=='frame_transfer.py' else original(p)):
                a=SpatialStore(root/'A',engine).engine_tag
            with patch.object(pipeline,'file_sha256',side_effect=lambda p:'version-B' if p.name=='frame_transfer.py' else original(p)):
                b=SpatialStore(root/'B',engine).engine_tag
        self.assertNotEqual(a,b)

    def test_helper_source_changes_launcher_build(self):
        from spatial_build import build_tag
        original=Path.read_bytes
        with patch.object(Path,'read_bytes',lambda p:b'version-A' if p.name=='frame_transfer.py' else original(p)):
            a=build_tag(True)
        with patch.object(Path,'read_bytes',lambda p:b'version-B' if p.name=='frame_transfer.py' else original(p)):
            b=build_tag(True)
        self.assertNotEqual(a,b)
