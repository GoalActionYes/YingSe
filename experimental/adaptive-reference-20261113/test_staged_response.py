import importlib
import unittest
import torch
import torch.nn.functional as F


class StagedResponseTests(unittest.TestCase):
    def api(self):return importlib.import_module('staged_response')
    def test_control_total_matches_existing_loss_and_paired_addition_is_exact(self):
        p=torch.linspace(.1,.8,48).reshape(4,3,2,2).requires_grad_();gt=p.detach().flip(0)
        rgb=F.l1_loss(p,gt);edge=F.l1_loss(p[:,:,1:]-p[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(p[:,:,:,1:]-p[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]);expected=rgb+.1*edge
        a=self.api().staged_loss(p,gt,False);b=self.api().staged_loss(p,gt,True)
        self.assertTrue(torch.equal(a['total'],expected));self.assertTrue(torch.equal(b['total'],expected+.25*b['delta']));b['total'].backward();self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(a['delta'].item(),0);self.assertTrue(torch.equal(a['delta'],b['delta']))
    def test_shared_offset_has_zero_pair_term_and_invalid_mode_stops(self):
        gt=torch.tensor([.1,.3,.5,.7]).reshape(4,1,1,1).expand(4,3,2,2);r=self.api().staged_loss(gt+.05,gt,True);self.assertAlmostEqual(r['delta'].item(),0,places=6)
        with self.assertRaises(ValueError):self.api().staged_loss(gt,gt,'yes')
    def test_only_paired_candidate_passing_both_gates_can_advance(self):
        for paired in (False,True):
            for a in (False,True):
                for b in (False,True):
                    result=self.api().staged_gate(dict(proceed_to_independent_evaluation=a),dict(proceed_to_independent_evaluation=b),paired)
                    self.assertEqual(result['proceed_to_independent_evaluation'],paired and a and b)


if __name__=='__main__':unittest.main()
