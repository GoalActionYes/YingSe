import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch


class FakeEngine:
    device=torch.device('cpu')
    def prepare(self,a,b):
        from spatial_render import PreparedSpatial
        return PreparedSpatial(self,torch.ones((1,13,8,16,16)),{'test':True})


class SpatialStoreTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('spatial_store'))
        return importlib.import_module('spatial_store')
    def test_persistence_integrity_and_cube_rejection(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a.png';b=root/'b.png'
            Image.new('RGB',(80,60),(40,100,180)).save(a);Image.new('RGB',(80,60),(80,90,130)).save(b)
            store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='test-v1')
            x,y=store.add_media(a),store.add_media(b);record=store.analyze(x['id'],y['id'])
            self.assertEqual(store.get_mapping_record(record['id'])['diagnostics']['engine'],'spatial')
            self.assertEqual(tuple(store.get_prepared(record['id']).grid.shape),(1,13,8,16,16))
            with self.assertRaises(ValueError):store.save_project(record['id'],1,{'kind':'cube'})
            again=api.SpatialStore(root/'data',FakeEngine(),identity_tag='test-v1')
            self.assertEqual(tuple(again.get_prepared(record['id']).grid.shape),(1,13,8,16,16))
            cache=store.root/'cache'/(record['cache_key']+'.npz');cache.write_bytes(b'broken')
            with self.assertRaises((ValueError,OSError)):again.get_prepared(record['id'])
    def test_changed_engine_identity_invalidates_project(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'source.png';Image.new('RGB',(32,32)).save(image)
            store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='one');a=store.add_media(image);record=store.analyze(a['id'],a['id'])
            changed=api.SpatialStore(root/'data',FakeEngine(),identity_tag='two')
            with self.assertRaises(ValueError):changed.get_prepared(record['id'])

    def test_supported_rgba_gray_and_palette_inputs_decode_to_rgb(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='modes')
            for mode,color in [('RGBA',(20,80,160,140)),('L',93),('P',3)]:
                image=Image.new(mode,(31,19),color)
                if mode=='P':image.putpalette([v for i in range(256) for v in (i,255-i,70)])
                path=root/(mode+'.png');image.save(path);record=store.add_media(path)
                actual,_=store.decode(record);np.testing.assert_array_equal(actual,np.array(image.convert('RGB')))


if __name__=='__main__':unittest.main()
