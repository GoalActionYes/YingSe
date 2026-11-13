import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import hashlib
import threading
import io
import struct
import zipfile
from unittest.mock import patch


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

    def test_different_checkpoint_profiles_cannot_reuse_cached_grid(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'source.png';Image.new('RGB',(32,32)).save(image)
            first=FakeEngine();first.checkpoint_sha='old-fixed-checkpoint'
            second=FakeEngine();second.checkpoint_sha='new-fixed-checkpoint'
            store=api.SpatialStore(root/'data',first);a=store.add_media(image);record=store.analyze(a['id'],a['id'])
            changed=api.SpatialStore(root/'data',second)
            self.assertNotEqual(store.engine_tag,changed.engine_tag)
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

    def test_cached_strength_preview_does_not_wait_for_export_gpu_lock(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a.png';b=root/'b.png';rgb=np.full((40,50,3),80,np.uint8)
            Image.fromarray(rgb).save(a);Image.new('RGB',(40,50),(100,110,120)).save(b)
            store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='cache');x,y=store.add_media(a),store.add_media(b);m=store.analyze(x['id'],y['id'])
            store.preview_cache[(m['id'],rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest())]=np.full_like(rgb,100)
            values=[]
            worker=threading.Thread(target=lambda:values.append(store.render(m['id'],rgb,.5,preview=True)))
            with store.render_lock:
                worker.start();worker.join(timeout=1)
                finished=not worker.is_alive()
            worker.join(timeout=5)
            self.assertTrue(finished,'Cached preview waited for export GPU lock')
            np.testing.assert_array_equal(values[0],np.full_like(rgb,90))

    def test_cached_preview_still_rejects_corrupt_grid_and_changed_source(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a.png';b=root/'b.png';rgb=np.full((40,50,3),80,np.uint8)
            Image.fromarray(rgb).save(a);Image.new('RGB',(40,50),(100,110,120)).save(b)
            store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='integrity')
            x,y=store.add_media(a),store.add_media(b);m=store.analyze(x['id'],y['id'])
            store.preview_cache[(m['id'],rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest())]=np.full_like(rgb,100)
            cache=store.root/'cache'/(m['cache_key']+'.npz');original=cache.read_bytes()
            cache.write_bytes(b'corrupt cached grid')
            with self.assertRaises((ValueError,OSError)):store.render(m['id'],rgb,.5,preview=True)
            cache.write_bytes(original)
            Image.new('RGB',(50,40),(81,81,81)).save(x['path'],format='PNG')
            with self.assertRaises(ValueError):store.render(m['id'],rgb,.5,preview=True)

    def test_analyze_rebuilds_valid_zip_with_corrupt_grid_crc(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a.png';Image.new('RGB',(32,32),(20,70,100)).save(a)
            store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='crc');source=store.add_media(a)
            mapping=store.analyze(source['id'],source['id']);cache=store.root/'cache'/(mapping['cache_key']+'.npz')
            original=cache.read_bytes();buffer=io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(original)) as src,zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as dst:
                for name in src.namelist():dst.writestr(name,src.read(name))
            raw=bytearray(buffer.getvalue())
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:offset=archive.getinfo('grid.npy').header_offset
            name_length,extra_length=struct.unpack_from('<HH',raw,offset+26)
            raw[offset+30+name_length+extra_length+140]^=1
            cache.write_bytes(raw)
            with self.assertRaises(zipfile.BadZipFile):store.get_prepared(mapping['id'])
            rebuilt=store.analyze(source['id'],source['id'])
            self.assertFalse(rebuilt['cache_hit'])
            self.assertEqual(tuple(store.get_prepared(rebuilt['id']).grid.shape),(1,13,8,16,16))

    def test_identity_reuses_probes_but_tracks_source_decoder_and_algorithm(self):
        api=self.api()
        from reference_color_app.storage import Store
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);decoder=root/'ffmpeg';decoder.write_bytes(b'decoder-one')
            target={'id':'a'*32,'path':str(root/'target'),'sha256':'source-one','kind':'video'}
            reference={'id':'b'*32,'path':str(root/'reference'),'sha256':'reference-one','kind':'image'}
            store=api.SpatialStore(root/'data',FakeEngine(),identity_tag='probes')
            with patch.object(api.media,'probe_media',side_effect=lambda path:{'kind':'video' if path==target['path'] else 'image','width':100,'metadata_text':'volatile-address'} ) as probe,patch.object(api.media,'ffmpeg_path',return_value=decoder):
                expected,_=Store._identity(store,target,reference,0.)
                expected['spatial_engine']='probes';probe.reset_mock()
                first,key=store._identity(target,reference,0.);self.assertEqual(first,expected)
                second,_=store._identity(target,reference,0.);self.assertEqual(first,second)
                self.assertEqual(probe.call_count,2,'Unchanged verified sources were probed twice')
                target=dict(target,sha256='source-two');store._identity(target,reference,0.)
                self.assertEqual(probe.call_count,3)
                decoder.write_bytes(b'decoder-two');changed,_=store._identity(target,reference,0.)
                self.assertNotEqual(first['decoder']['ffmpeg_sha256'],changed['decoder']['ffmpeg_sha256'])
                self.assertGreater(probe.call_count,3)
                with patch.object(api.pipeline,'algorithm_fingerprint',return_value='new-code'):
                    updated,_=store._identity(target,reference,0.)
                    self.assertEqual(updated['codefingerprint'],'new-code')


if __name__=='__main__':unittest.main()
