import importlib
import importlib.util
import unittest
import numpy as np


class CorrespondenceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('real_style_correspondence'))
        return importlib.import_module('real_style_correspondence')
    def test_perfect_and_swapped_costs(self):
        api=self.api();cost=np.ones((4,4))-np.eye(4);r=api.correspondence(cost)
        self.assertEqual(r['row_top1'],1);self.assertEqual(r['assignment_fraction_correct'],1);self.assertEqual(r['optimal_assignments'],1)
        r=api.correspondence(np.roll(cost,1,axis=1));self.assertEqual(r['row_top1'],0);self.assertEqual(r['assignment_fraction_correct'],0)
    def test_uniform_ties_do_not_favor_identity(self):
        r=self.api().correspondence(np.ones((4,4)))
        self.assertEqual(r['row_top1'],.25);self.assertEqual(r['assignment_fraction_correct'],.25);self.assertEqual(r['optimal_assignments'],24);self.assertAlmostEqual(r['identity_optimum_fraction'],1/24)
    def test_bad_costs_rejected(self):
        for value in [np.ones((2,2)),np.full((4,4),np.nan),-np.ones((4,4))]:
            with self.assertRaises(ValueError):self.api().correspondence(value)
    def test_gt_differences_do_not_wrap_uint8(self):
        images=np.full((4,2,3,3),128,dtype=np.uint8);images[0]=130
        self.assertEqual(sorted(self.api().gt_differences(images)),[0.,0.,0.,2.,2.,2.])


if __name__=='__main__':unittest.main()
