import importlib.util
import importlib
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from PIL import Image
from test_spatial_store import FakeEngine


class SpatialJobTests(unittest.TestCase):
    def test_zero_png_export_and_cube_rejection(self):
        self.assertIsNotNone(importlib.util.find_spec('spatial_jobs'))
        api=importlib.import_module('spatial_jobs');from spatial_store import SpatialStore
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'source.png';pixels=np.random.default_rng(7).integers(0,256,(31,47,3),dtype=np.uint8);Image.fromarray(pixels).save(path)
            store=SpatialStore(root/'data',FakeEngine(),identity_tag='test');media=store.add_media(path);mapping=store.analyze(media['id'],media['id'])
            manager=api.SpatialJobs(store)
            with self.assertRaises(ValueError):manager.start_export(mapping['id'],media['id'],1,'cube')
            job=manager.start_export(mapping['id'],media['id'],0,'png');deadline=time.monotonic()+8
            while manager.get_job(job)['status'] in ['queued','running'] and time.monotonic()<deadline:time.sleep(.02)
            self.assertEqual(manager.get_job(job)['status'],'done')
            with Image.open(manager.result_file(job)) as image:np.testing.assert_array_equal(np.array(image),pixels)


if __name__=='__main__':unittest.main()
