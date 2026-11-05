import unittest
from unittest.mock import patch
from candidate_pipeline import analyze

class CandidatePipelineTests(unittest.TestCase):
 def test_disabled_calls_original_without_segmentation(self):
  sentinel=object()
  with patch('candidate_pipeline.baseline.analyze',return_value=sentinel) as original:
   self.assertIs(analyze('target','reference',.25,enabled=False),sentinel)
   original.assert_called_once_with('target','reference',.25)
 def test_nonboolean_flag_rejected_before_io(self):
  for value in ('false',1,None):
   with self.assertRaises(ValueError):analyze('target','reference',enabled=value)
 def test_invalid_time_rejected_before_io(self):
  for value in (True,-1,float('nan'),float('inf'),None):
   with self.assertRaises(ValueError):analyze('target','reference',value,enabled=True)

if __name__=='__main__':unittest.main()
