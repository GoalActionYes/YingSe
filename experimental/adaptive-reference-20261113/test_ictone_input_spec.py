import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from ictone_input_spec import read_rgb_input_spec


class InputSpecTest(unittest.TestCase):
    def test_only_two_rgb_files_with_exact_hashes(self):
        root=Path(__file__).resolve().parents[2]/'outputs/adaptive-reference-20261113'
        with tempfile.TemporaryDirectory(prefix='ictone-input-test-',dir=root) as temp:
            folder=Path(temp).resolve()
            self.assertTrue(folder.is_relative_to(root.resolve()))
            path=folder/'rgb.png'
            Image.new('RGB',(256,256),(11,22,33)).save(path)
            item=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            spec=dict(index='test',content=item,reference=item)
            p=folder/'spec.json'
            p.write_text(json.dumps(spec))
            index,images=read_rgb_input_spec(p,root)
            self.assertEqual(index,'test')
            self.assertEqual(images['content'].getpixel((0,0)),(11,22,33))
            spec['gt']=str(path)
            p.write_text(json.dumps(spec))
            with self.assertRaises(AssertionError):read_rgb_input_spec(p,root)
            del spec['gt']
            spec['content']={**item,'sha256':'0'*64}
            p.write_text(json.dumps(spec))
            with self.assertRaises(AssertionError):read_rgb_input_spec(p,root)


if __name__=='__main__':unittest.main()
