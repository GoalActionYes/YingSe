import unittest
import numpy as np
import torch
from paired_training import grouped_batches,paired_delta_loss


class PairedTrainingTests(unittest.TestCase):
    def test_stream_uses_each_style_once_with_same_content_pairs(self):
        rows=[dict(group_id=str(g),content=f'{g}.png',style_index=s) for s in range(4) for g in range(6)]
        batches=grouped_batches(rows,np.random.default_rng(17))
        self.assertEqual(sorted(i for batch in batches for i in batch),list(range(24)))
        self.assertEqual(len(batches),6)
        for batch in batches:
            self.assertEqual(len(batch),4)
            for a,b in [batch[:2],batch[2:]]:
                self.assertEqual(rows[a]['content'],rows[b]['content'])
                self.assertNotEqual(rows[a]['style_index'],rows[b]['style_index'])
        self.assertEqual(batches,grouped_batches(rows,np.random.default_rng(17)))
        with self.assertRaises(ValueError):grouped_batches(rows[:-1],np.random.default_rng(17))
        corrupted=[dict(r) for r in rows];corrupted[0]['content']='different.png'
        with self.assertRaises(ValueError):grouped_batches(corrupted,np.random.default_rng(17))

    def test_pair_difference_removes_shared_offset_but_penalizes_wrong_response(self):
        gt=torch.tensor([.1,.3,.5,.7]).reshape(4,1,1,1).expand(4,3,2,2)
        prediction=(gt+.05).clone().requires_grad_()
        self.assertAlmostEqual(float(paired_delta_loss(prediction,gt)),0,places=6)
        ignored=gt[[0,0,2,2]].clone().requires_grad_()
        loss=paired_delta_loss(ignored,gt)
        self.assertAlmostEqual(float(loss),.2,places=6)
        loss.backward();self.assertGreater(float(ignored.grad.abs().sum()),0)
        with self.assertRaises(ValueError):paired_delta_loss(gt[:3],gt[:3])


if __name__=='__main__':unittest.main()
