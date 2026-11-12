import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from learned_store import LearnedStore
from reference_color_app.contracts import Mapping


def identity_cube():
    axis=np.linspace(0,1,64,dtype=np.float32)
    return np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)


class FakeProvider:
    def __init__(self):self.calls=0
    def compile(self,target,reference):
        self.calls+=1
        rgb=identity_cube(); final=np.clip(rgb+[.02,0,-.02],0,1).astype(np.float32)
        return Mapping(np.stack([.2*rgb+.8*final,final]),dict(engine='model',provider='test double'))


class StoreChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.provider=FakeProvider()
        self.store=LearnedStore(self.root/'data',self.provider,identity_tag='test-model-v1')
        source=np.random.default_rng(22).integers(40,200,(32,48,3),dtype=np.uint8)
        Image.fromarray(source).save(self.root/'target.png')
        Image.fromarray(source+8).save(self.root/'reference.png')
        self.target=self.store.add_media(self.root/'target.png')
        self.reference=self.store.add_media(self.root/'reference.png')
    def tearDown(self):self.temp.cleanup()

    def test_same_decoded_image_bypasses_model(self):
        record=self.store.analyze_engine(self.target['id'],self.target['id'],0,'model')
        mapping=self.store.get_mapping(record['id'])
        np.testing.assert_array_equal(mapping.layers[1],identity_cube())
        self.assertEqual(self.provider.calls,0)

    def test_cache_method_isolation_and_restart(self):
        record=self.store.analyze_engine(self.target['id'],self.reference['id'],0,'model')
        self.assertFalse(record['cache_hit'])
        expected=self.store.get_mapping(record['id']).layers.copy()
        cached=self.store.analyze_engine(self.target['id'],self.reference['id'],0,'model')
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(self.provider.calls,1)
        restarted=LearnedStore(self.root/'data',None,identity_tag='test-model-v1')
        np.testing.assert_array_equal(restarted.get_mapping(record['id']).layers,expected)
        stale=LearnedStore(self.root/'data',None,identity_tag='test-model-v2')
        with self.assertRaises(ValueError):stale.get_mapping(record['id'])
        baseline=self.store.analyze_engine(self.target['id'],self.reference['id'],0,'baseline')
        self.assertNotEqual(baseline['cache_key'],record['cache_key'])

    def test_mutated_input_and_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):self.store.analyze_engine(self.target['id'],self.reference['id'],True,'model')
        with self.assertRaises(ValueError):self.store.analyze_engine(self.target['id'],self.reference['id'],0,'unknown')
        record=self.store.analyze_engine(self.target['id'],self.reference['id'],0,'model')
        Path(self.target['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.store.get_mapping(record['id'])


if __name__=='__main__':unittest.main()
