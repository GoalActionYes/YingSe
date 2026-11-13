import importlib
import importlib.util
import unittest
from collections import defaultdict
from pathlib import Path
from prepare_triplet_cohort import select_cohort


class MultiReferenceCohortTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('prepare_multiref_cohort'))
        return importlib.import_module('prepare_multiref_cohort')
    def test_groups_keep_original_pair_and_have_distinct_targets_and_references(self):
        api=self.api()
        rows=[dict(content=f'content_images/{i:05d}.png',reference=f'style_images/{(i+1)%2000:05d}_s{j}.png',gt=f'style_images/{i:05d}_s{j}.png') for i in range(2000) for j in range(4)]
        old,stats=select_cohort(rows,dict(train=8,validation=4,reserved=4))
        train,_=api.select_grouped(rows,old['train'],4,4,'train',stats['audited'])
        val,_=api.select_grouped(rows,old['validation'],2,2,'validation',stats['audited'])
        grouped=defaultdict(list)
        for row in train:grouped[row['group_id']].append(row)
        self.assertEqual(len(grouped),4);self.assertEqual(len(train),16)
        originals={r['content_family']:r for r in old['train']}
        for family,group in grouped.items():
            self.assertEqual(len({r['gt'] for r in group}),4);self.assertEqual(len({r['reference'] for r in group}),4)
            first=next(r for r in group if r['style_index']==0)
            self.assertEqual(first['index'],originals[family]['index'])
            self.assertTrue(all(Path(r['content']).stem==family for r in group))
        families=lambda data:{r[k] for r in data for k in ['content_family','reference_family']}
        self.assertFalse(families(train)&families(val))
        with self.assertRaises(ValueError):api.select_grouped(rows,old['train'],4,5,'train',stats['audited'])


if __name__=='__main__':unittest.main()
