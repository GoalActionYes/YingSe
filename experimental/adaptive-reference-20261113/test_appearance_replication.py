import importlib
import importlib.util
import unittest
import torch

class AppearanceReplicationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('appearance_replication'))
        return importlib.import_module('appearance_replication')
    def test_full_initial_state_matches_despite_unrelated_prior_rng(self):
        api=self.api();base=torch.nn.Linear(3,3).state_dict();models=[]
        for seed in [1,9876]:
            torch.manual_seed(seed);model=torch.nn.Linear(3,3);api.initialize_replication_model(model,base);models.append(model)
        a,b=[m.state_dict() for m in models];self.assertEqual(set(a),set(b))
        for key,value in a.items():torch.testing.assert_close(value,b[key],atol=0,rtol=0)
        for key,value in base.items():torch.testing.assert_close(a[key],value,atol=0,rtol=0)
    def test_both_heads_start_zero_and_preserve_dtype(self):
        api=self.api();model=torch.nn.Linear(3,3).double();base={k:v.clone() for k,v in model.state_dict().items()};api.initialize_replication_model(model,base)
        for adapter in [model.pairwise_adapter,model.reference_adapter]:
            self.assertEqual(adapter.layers[0].weight.dtype,torch.float64)
            self.assertEqual(int(torch.count_nonzero(adapter.layers[-1].weight)),0)
            self.assertEqual(int(torch.count_nonzero(adapter.layers[-1].bias)),0)
    def test_second_initialization_rejected_without_modifying_parameters(self):
        api=self.api();model=torch.nn.Linear(3,3);base=model.state_dict();api.initialize_replication_model(model,base);before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError):api.initialize_replication_model(model,base)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
