import unittest
from pathlib import Path
import prepare_triplet_cohort as cohort


class TripletExpansionTests(unittest.TestCase):
    def test_expansion_keeps_development_prefix_and_retires_used_reserved_families(self):
        rows=[dict(content=f'content_images/{i:05d}.png',reference=f'style_images/{(i+1)%2000:05d}_tone.png',gt=f'style_images/{i:05d}_tone.png') for i in range(2000)]
        original,_=cohort.select_cohort(rows,dict(train=8,validation=4,reserved=4))
        used={r[k] for r in original['reserved'] for k in ['content_family','reference_family']}
        expanded,_=cohort.select_cohort(rows,dict(train=16,validation=8,reserved=4),used)
        self.assertEqual(expanded['train'][:8],original['train'])
        self.assertEqual(expanded['validation'][:4],original['validation'])
        families={p:{r[k] for r in group for k in ['content_family','reference_family']} for p,group in expanded.items()}
        self.assertFalse(families['reserved']&used)
        for a,b in [('train','validation'),('train','reserved'),('validation','reserved')]:self.assertFalse(families[a]&families[b])
        for group in expanded.values():
            for row in group:self.assertEqual(cohort.family(row['gt']),Path(row['content']).stem)

    def test_insufficient_or_invalid_group_counts_fail_instead_of_shrinking(self):
        for limits in [dict(train=True,validation=1,reserved=1),dict(train=1,validation=0,reserved=1)]:
            with self.assertRaises(ValueError):cohort.select_cohort([],limits)
        with self.assertRaises(ValueError):cohort.select_cohort([],dict(train=1,validation=1,reserved=1))


if __name__=='__main__':unittest.main()
