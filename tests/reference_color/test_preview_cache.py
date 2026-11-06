import unittest
import numpy as np
from reference_color_app.preview import PreviewFrames

class PreviewCacheTests(unittest.TestCase):
    def test_reuses_frame_and_bounds_memory(self):
        cache=PreviewFrames(max_bytes=1000);calls=[]
        def load():calls.append(1);return np.zeros((10,10,3),np.uint8),{'actual_time':0}
        a=cache.get(('a',0),load,lambda x:'before')
        b=cache.get(('a',0),load,lambda x:'before')
        self.assertEqual(len(calls),1);self.assertIs(a,b)
        self.assertFalse(a[0].flags.writeable)
        for i in range(10):cache.get((str(i),0),load,lambda x:'before')
        self.assertLessEqual(cache.bytes_used,1000)

    def test_large_frames_not_retained_and_failed_load_not_cached(self):
        cache=PreviewFrames(max_bytes=10);calls=[]
        def load():calls.append(1);return np.zeros((10,10,3),np.uint8),{}
        for _ in range(2):cache.get('a',load,lambda x:'before')
        self.assertEqual(len(calls),2);self.assertEqual(cache.bytes_used,0)
        with self.assertRaises(ValueError):cache.get('b',lambda:(_ for _ in ()).throw(ValueError()),lambda x:'before')

if __name__=='__main__':unittest.main()
