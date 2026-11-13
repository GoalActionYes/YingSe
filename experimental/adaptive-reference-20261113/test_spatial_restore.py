import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError
from PIL import Image
from test_spatial_store import FakeEngine
from spatial_workbench import make_spatial_workbench,assets
from reference_color_app.storage import _json


class SpatialRestoreTests(unittest.TestCase):
    def test_updated_engine_restores_sources_and_settings_without_using_old_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'input.png';Image.new('RGB',(32,32),(40,70,100)).save(image)
            server=make_spatial_workbench(root/'data',engine=FakeEngine());store=server.store
            source=store.add_media(image);mapping=store.analyze(source['id'],source['id'])
            settings={'kind':'png','clone_enabled':False,'adjustments':{'brightness':20}}
            project=store.save_project(mapping['id'],.47,settings)
            _json(store.root/'latest-project.json',{'project_id':project['id']})
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}/api/projects/latest'
            try:
                store.engine_tag='updated-model'
                with urlopen(url) as response:restored=json.load(response)
                self.assertTrue(restored['needs_analysis']);self.assertIsNone(restored['mapping_id'])
                self.assertEqual(restored['target']['id'],source['id']);self.assertEqual(restored['reference']['id'],source['id'])
                self.assertAlmostEqual(restored['strength'],.47);self.assertEqual(restored['output_settings'],settings)
                fresh=store.analyze(source['id'],source['id']);saved=store.save_project(fresh['id'],.47,settings)
                _json(store.root/'latest-project.json',{'project_id':saved['id']})
                with urlopen(url) as response:valid=json.load(response)
                self.assertFalse(valid['needs_analysis']);self.assertEqual(valid['mapping_id'],fresh['id'])
                Image.new('RGB',(32,32),(90,90,90)).save(source['path'],format='PNG')
                with self.assertRaises(HTTPError) as error:urlopen(url)
                self.assertEqual(error.exception.code,400)
                js=assets()['/app.js'][0].decode()
                self.assertIn('if(project.needs_analysis)',js)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
