import unittest
import numpy as np
from multiref_response import response_metrics


class MultiReferenceResponseTests(unittest.TestCase):
    def test_correct_reference_change_beats_ignoring_and_reversing_it(self):
        a=np.full((2,2,3),80,dtype=np.uint8);b=a+40
        correct=response_metrics(a,b,a,b)
        ignored=response_metrics(a,a,a,b)
        reversed_order=response_metrics(b,a,a,b)
        self.assertEqual(correct['delta_mae'],0)
        self.assertAlmostEqual(correct['response_gain'],1)
        self.assertEqual(ignored['response_gain'],0)
        self.assertEqual(ignored['delta_mae'],40)
        self.assertEqual(reversed_order['delta_mae'],80)
        self.assertAlmostEqual(reversed_order['response_gain'],-1)

    def test_zero_change_has_no_defined_direction_and_bad_inputs_fail(self):
        a=np.zeros((2,2,3),dtype=np.uint8)
        result=response_metrics(a,a,a,a)
        self.assertIsNone(result['response_gain'])
        self.assertEqual(result['gt_change_mae'],0)
        for bad in [a[:1],np.full(a.shape,np.nan),np.full(a.shape,256.)]:
            with self.assertRaises(ValueError):response_metrics(a,bad,a,a)


if __name__=='__main__':unittest.main()
