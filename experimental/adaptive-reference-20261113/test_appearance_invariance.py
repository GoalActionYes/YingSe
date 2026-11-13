import importlib
import importlib.util
import unittest


def table(mae,delta,tone):
    keys=['brightness','contrast','saturation','light_minus_dark_cb','light_minus_dark_cr']
    return dict(mean=mae,delta_mean=delta,cases=[dict(index=i,mae=mae) for i in range(64)],paired=[dict(group_id=str(i),delta_mae=delta,tones=dict(absolute_delta_error={k:tone for k in keys})) for i in range(32)])

class InvarianceGateTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('appearance_invariance'))
        return importlib.import_module('appearance_invariance')
    def test_primary_and_strong_appearance_nonregression_both_required(self):
        api=self.api();baseline=table(20,20,10);c00=table(19,19,9);c10=table(17,17,8.5);c11=table(16,16,8)
        gate=api.invariance_gate(baseline,c11,c00,c10)
        self.assertTrue(gate['primary_gate_passed']);self.assertTrue(gate['proceed_to_independent_evaluation'])
    def test_better_than_c10_cannot_override_failed_original_gate(self):
        api=self.api();gate=api.invariance_gate(table(20,20,10),table(25,25,12),table(19,19,9),table(30,30,15))
        self.assertFalse(gate['primary_gate_passed']);self.assertFalse(gate['proceed_to_independent_evaluation'])
    def test_rgb_and_reference_delta_nonregression_checked_separately(self):
        api=self.api();base,candidate,c00=table(20,20,10),table(16,16,8),table(19,19,9)
        for c10,key in [(table(15,17,8.5),'rgb_no_worse'),(table(17,15,8.5),'delta_no_worse')]:
            gate=api.invariance_gate(base,candidate,c00,c10)
            self.assertTrue(gate['primary_gate_passed']);self.assertFalse(gate['c10_nonregression'][key]);self.assertFalse(gate['proceed_to_independent_evaluation'])
    def test_secondary_uses_case_and_pair_values_not_forged_summary(self):
        api=self.api();base,candidate,c00,c10=table(20,20,10),table(16,16,8),table(19,19,9),table(15,15,8.5)
        c10['mean']=999;c10['delta_mean']=999
        gate=api.invariance_gate(base,candidate,c00,c10)
        self.assertFalse(gate['proceed_to_independent_evaluation'])
        self.assertEqual(gate['c10_nonregression']['appearance_mean'],15);self.assertEqual(gate['c10_nonregression']['appearance_delta_mean'],15)

if __name__=='__main__':unittest.main()
