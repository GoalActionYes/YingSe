import importlib
import importlib.util
import unittest
import torch
from torch import nn


class FinalCheckpointTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('canonical_final'))
        return importlib.import_module('canonical_final')

    def test_exact_finite_inference_state_loads(self):
        api=self.api(); model=nn.Linear(3,2)
        state={k:torch.full_like(v,.25) for k,v in model.state_dict().items()}
        api.load_final_state(model,{'epoch':1,'model_state_dict':state})
        for value in model.state_dict().values(): torch.testing.assert_close(value,torch.full_like(value,.25))

    def test_missing_extra_shape_dtype_and_nonfinite_state_fail(self):
        api=self.api(); model=nn.Linear(3,2)
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        bad=[{'weight':state['weight']},dict(state,unexpected=torch.zeros(1)),dict(state,weight=torch.zeros(2,2)),
             dict(state,weight=state['weight'].double()),dict(state,bias=torch.full((2,),float('nan')))]
        for candidate in bad:
            with self.subTest(keys=list(candidate)),self.assertRaises(ValueError):
                api.load_final_state(model,{'model_state_dict':candidate})


if __name__=='__main__': unittest.main()
