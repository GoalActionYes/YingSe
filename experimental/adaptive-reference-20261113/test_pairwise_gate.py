import copy
import unittest
from pairwise_gate import response_development_gate,require_completed_control


def scores(mae,delta,relationship=10,brightness=5,saturation=5,contrast=5):
    return dict(cases=[dict(index=i,mae=mae) for i in range(64)],paired=[dict(group_id=str(i),delta_mae=delta,
        tones=dict(absolute_delta_error=dict(brightness=brightness,saturation=saturation,contrast=contrast,light_minus_dark_cb=relationship,light_minus_dark_cr=relationship))) for i in range(32)])


class PairwiseGateTests(unittest.TestCase):
    def test_training_flag_alone_cannot_be_used_as_a_completed_control(self):
        config=dict(epochs=8,max_updates=2048);source={'train.py':'a'*64}
        record=dict(finished=True,config=config,source_sha256=source,epochs=[dict(epoch=i,updates=i*256) for i in range(1,9)],best_epoch=8)
        with self.assertRaises(ValueError):require_completed_control(record,config,source)
        record.update(development_gate={'proceed_to_independent_evaluation':False},selected_checkpoint_sha256='b'*64)
        require_completed_control(record,config,source)
        with self.assertRaises(ValueError):require_completed_control(record,config,{'train.py':'c'*64})
        record['epochs'][-1]['updates']=2000
        with self.assertRaises(ValueError):require_completed_control(record,config,source)

    def test_rgb_improvement_cannot_hide_worse_luminance_color_relationship(self):
        gate=response_development_gate(scores(20,10),scores(14,8,11))
        self.assertFalse(gate['proceed_to_independent_evaluation'])

    def test_candidate_must_add_relationship_benefit_over_matched_control(self):
        gate=response_development_gate(scores(20,12),scores(14,8,8),scores(16,10,7))
        self.assertFalse(gate['proceed_to_independent_evaluation'])
        self.assertTrue(response_development_gate(scores(20,12),scores(14,8,6),scores(16,10,8))['proceed_to_independent_evaluation'])

    def test_saturation_regression_or_incomplete_pairing_is_not_hidden(self):
        self.assertFalse(response_development_gate(scores(20,10),scores(14,8,8,saturation=6))['proceed_to_independent_evaluation'])
        bad=copy.deepcopy(scores(14,8,8));bad['paired'].pop()
        with self.assertRaises(ValueError):response_development_gate(scores(20,10),bad)


if __name__=='__main__':unittest.main()
