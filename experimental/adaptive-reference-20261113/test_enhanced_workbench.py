import json
from pathlib import Path
import tempfile
import unittest
from spatial_workbench import make_spatial_workbench
from test_spatial_store import FakeEngine
from spatial_build import build_tag


class EnhancedWorkbenchTests(unittest.TestCase):
    def test_enhanced_instance_is_separate_and_enables_only_its_video_smoothing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old=make_spatial_workbench(root/'old',engine=FakeEngine())
            enhanced=make_spatial_workbench(root/'enhanced',engine=FakeEngine(),enhanced=True)
            try:
                self.assertFalse(old.manager.temporal_enabled)
                self.assertTrue(enhanced.manager.temporal_enabled)
                self.assertNotEqual(old.build_tag,enhanced.build_tag)
                self.assertEqual(enhanced.build_tag,build_tag(enhanced=True))
                self.assertIn('自适应参考调色',enhanced.assets['/'][0].decode())
                self.assertNotEqual(old.store.root,enhanced.store.root)
            finally:old.server_close();enhanced.server_close()


if __name__=='__main__':unittest.main()
