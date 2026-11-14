import importlib
import unittest
import numpy as np


class CoverageTrainingTests(unittest.TestCase):
    def rows(self,groups):return [dict(group_id=f'g{g}',content=f'c{g}',style_index=s,index=g*4+s) for s in range(4) for g in range(groups)]
    def api(self):return importlib.import_module('coverage_training')
    def test_equal_updates_keep_complete_passes_and_real_content_pairs(self):
        api=self.api();small=self.rows(4);large=self.rows(16)
        a=api.block_batches(small,np.random.default_rng(24),4);b=api.block_batches(large,np.random.default_rng(24),1)
        self.assertEqual(len(a),len(b));self.assertEqual(len(a),16)
        self.assertEqual(a,api.block_batches(small,np.random.default_rng(24),4))
        for rows,batches,passes in [(small,a,4),(large,b,1)]:
            counts=np.bincount([i for batch in batches for i in batch],minlength=len(rows))
            self.assertTrue(np.all(counts==passes))
            for batch in batches:
                self.assertEqual(len(set(batch)),4)
                self.assertEqual(rows[batch[0]]['group_id'],rows[batch[1]]['group_id'])
                self.assertEqual(rows[batch[2]]['group_id'],rows[batch[3]]['group_id'])
                self.assertNotEqual(rows[batch[0]]['group_id'],rows[batch[2]]['group_id'])
    def test_invalid_pass_count_or_incomplete_groups_stop(self):
        api=self.api()
        for passes in [0,True,1.5]:
            with self.assertRaises(ValueError):api.block_batches(self.rows(4),np.random.default_rng(24),passes)
        with self.assertRaises(ValueError):api.block_batches(self.rows(4)[:-1],np.random.default_rng(24),4)
    def test_only_expanded_candidate_passing_both_gates_can_advance(self):
        api=self.api()
        for expanded in (False,True):
            for a in (False,True):
                for b in (False,True):
                    matched=dict(proceed_to_independent_evaluation=a,metric=123)
                    result=api.combined_gate(matched,dict(proceed_to_independent_evaluation=b),expanded)
                    self.assertEqual(result['proceed_to_independent_evaluation'],expanded and a and b)
                    self.assertEqual(matched,dict(proceed_to_independent_evaluation=a,metric=123))
        with self.assertRaises(ValueError):api.combined_gate(dict(proceed_to_independent_evaluation='true'),dict(proceed_to_independent_evaluation=True),True)


if __name__=='__main__':unittest.main()
