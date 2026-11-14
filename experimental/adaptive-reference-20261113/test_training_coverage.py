import copy
import importlib
import unittest
from collections import defaultdict
from prepare_triplet_cohort import select_cohort
from prepare_multiref_cohort import select_grouped


class TrainingCoverageTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(content=f'content/{i:05d}.png',reference=f'style/{(i+1)%1000:05d}_s{j}.png',gt=f'style/{i:05d}_s{j}.png') for i in range(1000) for j in range(4)]
        first,stats=select_cohort(rows,dict(train=8,validation=4,reserved=4))
        old,_=select_grouped(rows,first['train'],4,4,'train',stats['audited'])
        held=[dict(groups=dict(validation=first['validation'],reserved=first['reserved']))]
        return rows,old,held,stats['audited']

    def api(self):return importlib.import_module('training_coverage')

    def test_keeps_old_rows_and_deterministically_adds_complete_groups(self):
        api=self.api();rows,old,held,audited=self.fixture();snapshot=copy.deepcopy((rows,old,held))
        result,stats=api.expand_training(rows,old,held,8,audited)
        again,other=api.expand_training(rows,old,held,8,audited)
        self.assertEqual((result,stats),(again,other));self.assertEqual((rows,old,held),snapshot)
        by=defaultdict(list)
        for r in result:by[r['group_id']].append(r)
        self.assertEqual(len(by),8);self.assertEqual(len(result),32)
        self.assertEqual([r for r in result if r['group_id'] in {v['group_id'] for v in old}],old)
        for name,group in by.items():
            self.assertEqual([r['style_index'] for r in group],list(range(4)))
            self.assertEqual(len({r['gt'] for r in group}),4);self.assertEqual(len({r['reference'] for r in group}),4)
            self.assertTrue(all(r['content_family']==name for r in group))
        excluded={r[k] for h in held for part in h['groups'].values() for r in part for k in ['content_family','reference_family']}
        self.assertFalse(excluded & {r[k] for r in result for k in ['content_family','reference_family']})
        self.assertEqual(stats['old_groups'],4);self.assertEqual(stats['added_groups'],4)

    def test_reference_role_exclusion_is_honored_even_for_train_family(self):
        api=self.api();rows,old,held,audited=self.fixture()
        initial,_=api.expand_training(rows,old,held,8,audited)
        oldnames={r[k] for r in old for k in ['content_family','reference_family']}
        target=next(r['reference_family'] for r in initial if r['reference_family'] not in oldnames)
        held.append(dict(groups=dict(reserved=[dict(content_family='unused',reference_family=target)])))
        result,_=api.expand_training(rows,old,held,8,audited)
        self.assertNotIn(target,{r[k] for r in result for k in ['content_family','reference_family']})

    def test_refuses_to_rewrite_old_groups_or_admit_holdout(self):
        api=self.api();rows,old,held,audited=self.fixture()
        bad=copy.deepcopy(old);bad[0]['gt']='style/99999_s0.png'
        with self.assertRaises(ValueError):api.expand_training(rows,bad,held,8,audited)
        held.append(dict(groups=dict(validation=[dict(content_family=old[0]['content_family'],reference_family='unused')])))
        with self.assertRaises(ValueError):api.expand_training(rows,old,held,8,audited)

    def test_rejects_ambiguous_families_and_changed_gt_convention(self):
        api=self.api();rows,old,held,audited=self.fixture();bad=copy.deepcopy(rows)
        bad[-1]['content']='other/'+bad[-1]['content'].split('/')[-1]
        with self.assertRaises(ValueError):api.expand_training(bad,old,held,8,audited)
        bad=copy.deepcopy(rows);bad[-1]['gt']='style/99999_s0.png'
        with self.assertRaises(ValueError):api.expand_training(bad,old,held,8,audited)

    def test_invalid_count_and_insufficient_data_do_not_shrink(self):
        api=self.api();rows,old,held,audited=self.fixture()
        for n in [True,0,3,100000]:
            with self.assertRaises(ValueError):api.expand_training(rows,old,held,n,audited)


if __name__=='__main__':unittest.main()
