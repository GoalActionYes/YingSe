import json
from pathlib import Path
import tempfile
import unittest
import base64
import io
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from PIL import Image
import numpy as np
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

    def test_verified_basis_matches_normal_preview_and_rejects_invalid_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'same.png';Image.new('RGB',(33,21),(30,90,170)).save(path)
            server=make_spatial_workbench(root/'data',engine=FakeEngine(),enhanced=True)
            source=server.store.add_media(path);mapping=server.store.analyze(source['id'],source['id'])
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            def preview(flag):
                data=dict(mapping_id=mapping['id'],strength=.37,adjustments={'basic':{'brightness':12}},interactive=True,include_basis=flag)
                with urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/preview',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})) as response:return json.load(response)
            try:
                full=preview(True);normal=preview(False)
                self.assertEqual(full['after'],normal['after']);self.assertNotIn('basis',normal)
                self.assertEqual((full['basis']['width'],full['basis']['height']),(33,21))
                def pixels(uri):return np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))))
                np.testing.assert_array_equal(pixels(full['basis']['full']),pixels(full['before']))
                with self.assertRaises(HTTPError) as error:preview('invalid')
                self.assertEqual(error.exception.code,400)
                (server.store.root/'cache'/(mapping['cache_key']+'.npz')).write_bytes(b'broken')
                for flag in [False,True]:
                    with self.assertRaises(HTTPError) as error:preview(flag)
                    self.assertEqual(error.exception.code,400)
            finally:server.shutdown();server.server_close();worker.join(timeout=5)


if __name__=='__main__':unittest.main()
