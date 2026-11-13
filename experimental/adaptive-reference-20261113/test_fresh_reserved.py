import unittest
from prepare_fresh_reserved import select_fresh_reserved
from prepare_triplet_cohort import select_cohort


class FreshReservedTests(unittest.TestCase):
    def test_all_consumed_and_development_families_are_excluded(self):
        rows=[dict(content=f'content_images/{i:05d}.png',reference=f'style_images/{(i+1)%4000:05d}_a.png',gt=f'style_images/{i:05d}_a.png') for i in range(4000)]
        limits=dict(train=4,validation=4,reserved=4)
        first,_=select_cohort(rows,limits)
        families=lambda rs:{r[k] for r in rs for k in ['content_family','reference_family']}
        second,_=select_cohort(rows,limits,families(first['reserved']))
        provisional,_=select_fresh_reserved(rows,[{'groups':first},{'groups':second}],{'groups':dict(train=[],validation=[])},4)
        development={'groups':dict(train=provisional[:1],validation=[])}
        selected,stats=select_fresh_reserved(rows,[{'groups':first},{'groups':second}],development,4)
        excluded=families(first['reserved']+second['reserved']+provisional[:1])
        self.assertFalse(families(selected)&excluded)
        self.assertEqual(len({r['content_family'] for r in selected}),4)
        self.assertEqual(selected,select_fresh_reserved(rows,[{'groups':first},{'groups':second}],development,4)[0])
        self.assertTrue(excluded.issubset(set(stats['excluded_families'])))


if __name__=='__main__':unittest.main()
