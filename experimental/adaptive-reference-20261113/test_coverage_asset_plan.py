import copy
import importlib
import unittest


class CoverageAssetPlanTests(unittest.TestCase):
    def api(self):return importlib.import_module('coverage_asset_plan')
    def record(self,path='content/a.png',size=100):return dict(path=path,size=size,oid='a'*40,lfs=dict(oid='b'*64))

    def test_exact_set_and_both_git_lfs_identities_are_required(self):
        api=self.api();rows=[self.record()];saved=copy.deepcopy(rows)
        self.assertEqual(api.validate_metadata(rows,{'content/a.png'},100),100);self.assertEqual(rows,saved)
        for bad in [[],rows+rows,[self.record('other.png')],[dict(rows[0],oid='z'*40)],[dict(rows[0],lfs=dict(oid='z'*64))]]:
            with self.assertRaises(ValueError):api.validate_metadata(bad,{'content/a.png'},100)
        git=[dict(rows[0])];del git[0]['lfs']
        self.assertEqual(api.validate_metadata(git,{'content/a.png'},100),100)

    def test_unsafe_paths_and_byte_overruns_stop_without_truncation(self):
        api=self.api()
        for path in ['../a.png','a/../a.png','/a.png','C:/a.png','a\\b.png','./a.png','a//b.png','']:
            with self.assertRaises(ValueError):api.validate_metadata([self.record(path)],{path},1000)
        for size in [True,0,-1,20*1024**2+1]:
            with self.assertRaises(ValueError):api.validate_metadata([self.record(size=size)],{'content/a.png'},10**9)
        with self.assertRaises(ValueError):api.validate_metadata([self.record()],{'content/a.png'},99)


if __name__=='__main__':unittest.main()
