import unittest
from pathlib import Path
from triplet_dataset_options import dataset_options


class DatasetOptionsTests(unittest.TestCase):
    def test_existing_modes_and_multiref_have_explicit_bounds_and_reuse_order(self):
        root=Path('root')
        original=dataset_options(root,False,False)
        expanded=dataset_options(root,True,False)
        grouped=dataset_options(root,False,True)
        self.assertEqual(original['directory'].name,'cohort-v1')
        self.assertEqual(original['max_bytes'],600*1024**2)
        self.assertEqual(original['reuse'],[])
        self.assertEqual(expanded['directory'].name,'cohort-v2')
        self.assertEqual([p.name for p in expanded['reuse']],['cohort-v1'])
        self.assertEqual(grouped['directory'].name,'cohort-multiref-v1')
        self.assertEqual(grouped['max_bytes'],2*1024**3)
        self.assertEqual([p.name for p in grouped['reuse']],['cohort-v2','cohort-v1'])
        with self.assertRaises(ValueError):dataset_options(root,True,True)


if __name__=='__main__':unittest.main()
