import importlib
import importlib.util
import unittest


class TrainingGateTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training_gates'))
        return importlib.import_module('training_gates')
    def test_new_validation_cannot_regress_behind_good_aggregate(self):
        api=self.api()
        baseline={'cases':[{'index':i,'mae':100. if i<32 else 10.} for i in range(64)]}
        selected={'cases':[{'index':i,'mae':20. if i<32 else 9.7 if i<52 else 15.} for i in range(64)]}
        gate=api.development_gate(baseline,selected,expanded=True)
        self.assertGreater(gate['relative_improvement'],.5)
        self.assertEqual(gate['improved'],52)
        self.assertEqual(gate['severe_regressions'],0)
        self.assertFalse(gate['proceed_to_independent_evaluation'])
        self.assertGreater(gate['added_validation']['selected_mean'],gate['added_validation']['baseline_mean'])
    def test_joint_candidate_must_beat_matched_control(self):
        api=self.api()
        make=lambda value:{'cases':[{'index':i,'mae':value} for i in range(64)]}
        self.assertTrue(api.development_gate(make(20),make(15),expanded=True)['proceed_to_independent_evaluation'])
        self.assertFalse(api.development_gate(make(20),make(15),expanded=True,control=make(14))['proceed_to_independent_evaluation'])
    def test_incomplete_duplicate_and_nonfinite_scores_rejected(self):
        api=self.api();valid={'cases':[{'index':i,'mae':10.} for i in range(64)]}
        for cases in [valid['cases'][:63],[{'index':0,'mae':9.}]*64,[{'index':i,'mae':float('nan') if i==2 else 9.} for i in range(64)]]:
            with self.assertRaises(ValueError):api.development_gate(valid,{'cases':cases},expanded=True)

    def test_independent_candidate_must_beat_current_app_not_only_legacy_global(self):
        api=self.api()
        rows=[{'index':i,'scores':{'current':30.,'before':10.,'candidate':12.}} for i in range(32)]
        self.assertFalse(api.reserved_gate(rows)['passed'])
    def test_independent_mean_gain_cannot_hide_two_severe_regressions(self):
        api=self.api()
        rows=[{'index':i,'scores':{'current':30.,'before':10.,'candidate':5. if i<30 else 20.}} for i in range(32)]
        gate=api.reserved_gate(rows)
        self.assertGreater(gate['primary']['relative_improvement'],.3)
        self.assertEqual(gate['primary']['severe'],2)
        self.assertFalse(gate['passed'])
    def test_independent_requires_all_cases_and_no_exact_overlap(self):
        api=self.api()
        rows=[{'index':i,'scores':{'current':30.,'before':20.,'candidate':14.}} for i in range(32)]
        self.assertTrue(api.reserved_gate(rows)['passed'])
        self.assertFalse(api.reserved_gate(rows[:-1])['passed'])
        self.assertFalse(api.reserved_gate(rows,duplicates=['overlap'])['passed'])


if __name__=='__main__':unittest.main()
