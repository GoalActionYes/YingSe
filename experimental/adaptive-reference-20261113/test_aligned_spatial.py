import tempfile
import unittest
import time
from pathlib import Path
import numpy as np
from PIL import Image
from test_spatial_store import FakeEngine
from spatial_store import SpatialStore
from spatial_render import finish_spatial


class AlignedSpatialTests(unittest.TestCase):
    def test_shift_and_local_content_change_rejected(self):
        from reference_color_app.aligned_reference import fit_aligned
        source=np.random.default_rng(12).integers(30,210,(96,96,3),dtype=np.uint8)
        shifted=np.roll(source,2,axis=1)
        changed=source.copy();changed[:32,:32]=255-changed[:32,:32]
        for reference in [shifted,changed]:
            self.assertFalse(fit_aligned(source,reference)['accepted'])

    def test_real_video_export_keeps_selected_nonzero_frame_mapping(self):
        from independent_coco import ROOT
        from spatial_jobs import SpatialJobs
        source=ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
        if not source.exists():self.skipTest('Local public video fixture unavailable')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);engine=FakeEngine();engine.profile='multiref-v1'
            store=SpatialStore(root/'data',engine);media=store.add_media(source)
            frame,_=store.decode(media,1.)
            reference=np.clip(frame.astype(int)+[12,0,-12],0,255).astype(np.uint8)
            path=root/'reference.png';Image.fromarray(reference).save(path);ref=store.add_media(path)
            record=store.analyze(media['id'],ref['id'],1.)
            self.assertTrue(record['diagnostics']['certified_aligned']['accepted'])
            manager=SpatialJobs(store,temporal=True)
            job=manager.start_export(record['id'],media['id'],.5,'mp4',{'basic':{'brightness':10}})
            deadline=time.monotonic()+40
            while manager.get_job(job)['status'] in ['queued','running'] and time.monotonic()<deadline:time.sleep(.05)
            result=manager.get_job(job)
            if result['status'] in ['queued','running']:
                manager.cancel_job(job)
                self.fail('Owned fixed export did not finish within timeout')
            self.assertEqual(result['status'],'done',result.get('error'))
            self.assertEqual(result['settings']['frames'],90)
            self.assertIsNone(result['settings']['temporal'])
            self.assertEqual(result['settings']['algorithm'],'Certified same-frame fixed affine LUT')

    def test_different_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rng=np.random.default_rng(7);paths=[]
            for name in ['a','b']:
                p=root/(name+'.png');Image.fromarray(rng.integers(30,210,(64,64,3),dtype=np.uint8)).save(p);paths.append(p)
            engine=FakeEngine();engine.profile='multiref-v1';store=SpatialStore(root/'data',engine)
            a,b=[store.add_media(p) for p in paths];r=store.analyze(a['id'],b['id'])
            self.assertFalse(r['diagnostics']['certified_aligned']['accepted'])

    def test_video_uses_fixed_mapping_without_neural_reanalysis(self):
        from aligned_spatial import PreparedAligned
        from reference_color_app.aligned_reference import fit_aligned,make_mapping
        from spatial_jobs import grade_video_frame
        rng=np.random.default_rng(9);source=rng.integers(30,210,(64,64,3),dtype=np.uint8)
        reference=(source.astype(int)+[12,0,-12]).astype(np.uint8)
        class NoAnalysis:
            def prepare(self,*args):raise AssertionError('Fixed video unexpectedly invoked neural model')
        fixed=PreparedAligned(NoAnalysis(),None,{},make_mapping(fit_aligned(source,reference)))
        frame=np.roll(source,1,axis=1);settings={'basic':{'contrast':12}}
        actual=grade_video_frame(fixed.engine,frame,reference,.5,settings,fixed_prepared=fixed)
        np.testing.assert_array_equal(actual,finish_spatial(frame,fixed.apply(frame,1),.5,settings))
        # A later frame can equal the selected reference without changing route.
        actual=grade_video_frame(fixed.engine,reference,reference,1,None,fixed_prepared=fixed)
        np.testing.assert_array_equal(actual,fixed.apply(reference,1))

    def test_certified_mapping_survives_reload_and_manual_mix(self):
        rng=np.random.default_rng(5)
        source=rng.integers(30,210,(64,64,3),dtype=np.uint8)
        truth=(source.astype(int)+[12,0,-12]).astype(np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.png';b=root/'b.png'
            Image.fromarray(source).save(a);Image.fromarray(truth).save(b)
            engine=FakeEngine();engine.profile='multiref-v1'
            store=SpatialStore(root/'data',engine);x,y=store.add_media(a),store.add_media(b)
            record=store.analyze(x['id'],y['id'])
            self.assertTrue(record['diagnostics'].get('certified_aligned',{}).get('accepted'))
            again=SpatialStore(root/'data',engine)
            prepared=again.get_prepared(record['id'])
            full=prepared.apply(source,1)
            self.assertLess(np.abs(full.astype(float)-truth).mean(),.1)
            settings={'basic':{'brightness':15}}
            np.testing.assert_array_equal(again.render(record['id'],source,.5,settings),finish_spatial(source,full,.5,settings))
            (again.root/'cache'/(record['cache_key']+'.npz')).write_bytes(b'broken')
            with self.assertRaises((ValueError,OSError)):again.get_prepared(record['id'])

    def test_app8_does_not_enable_new_route(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'a.png';Image.fromarray(np.random.default_rng(4).integers(0,256,(32,32,3),dtype=np.uint8)).save(p)
            store=SpatialStore(root/'data',FakeEngine());x=store.add_media(p)
            record=store.analyze(x['id'],x['id'])
            self.assertFalse(record['diagnostics'].get('certified_aligned',{}).get('accepted',False))
