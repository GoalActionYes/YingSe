import importlib
import importlib.util
import unittest
import torch


class ReferencePatchTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('reference_patch'))
        return importlib.import_module('reference_patch')

    def fixture(self):
        api=self.api();torch.manual_seed(13);module=api.ReferencePatchContext().double();query=torch.randn(2,128,2,3,dtype=torch.float64);tokens=torch.randn(2,16,128,dtype=torch.float64)
        return api,module,query,tokens

    def unzero(self,module):
        with torch.no_grad():module.proj.weight.copy_(torch.eye(128,dtype=torch.float64));module.proj.bias.zero_()

    def test_zero_residual_parameter_count_and_initial_projection_gradient(self):
        _,module,query,tokens=self.fixture();self.assertEqual(sum(v.numel() for v in module.parameters()),65664)
        output=module(query,tokens);self.assertTrue(torch.equal(output,torch.zeros_like(query)))
        output.sum().backward();self.assertGreater(module.proj.weight.grad.abs().max().item(),0);self.assertEqual(module.q.weight.grad.abs().max().item(),0)

    def test_nonconstant_tokens_enable_query_dependence_and_key_gradients(self):
        _,module,query,tokens=self.fixture();self.unzero(module);query.requires_grad_();a=module(query,tokens);b=module(-query,tokens)
        self.assertGreater((a-b).abs().max().item(),1e-8);a.square().sum().backward()
        for g in [query.grad,module.q.weight.grad,module.kv.weight.grad[:128]]:self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0)

    def test_pooled_control_is_query_invariant(self):
        api,module,query,_=self.fixture();self.unzero(module);features=torch.randn(2,128,16,16,dtype=torch.float64);tokens=api.reference_tokens(features,pooled=True)
        self.assertTrue(torch.equal(tokens,tokens[:,:1].expand_as(tokens)));a=module(query,tokens);b=module(-query,tokens)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=0));self.assertTrue(torch.allclose(a,a[:,:,:1,:1].expand_as(a),atol=1e-12,rtol=0))

    def test_token_permutation_invariance_and_pool_shapes(self):
        api,module,query,tokens=self.fixture();self.unzero(module);order=torch.randperm(16)
        self.assertTrue(torch.allclose(module(query,tokens),module(query,tokens[:,order]),atol=1e-12,rtol=0))
        self.assertEqual(api.reference_tokens(torch.randn(2,128,16,16)).shape,(2,16,128))

    def test_bad_shapes_types_and_nonfinite_inputs_rejected(self):
        api,module,query,tokens=self.fixture()
        for bad in [tokens[:,:15],tokens[:1],torch.full_like(tokens,float('nan')),tokens.float()]:
            with self.assertRaises(ValueError):module(query,bad)
        for bad in [torch.ones(2,127,4,4),torch.ones(2,128,4),torch.full((2,128,4,4),float('inf'))]:
            with self.assertRaises(ValueError):api.reference_tokens(bad)
        with self.assertRaises(ValueError):api.reference_tokens(torch.ones(2,128,4,4),pooled=1)

    def test_both_full_control_gates_required(self):
        from test_appearance_invariance import table
        api=self.api();baseline=table(20,20,10);selected=table(15,15,8);good=table(18,18,9);strong=table(14,14,7)
        result=api.patch_development_gate(baseline,selected,good,good)
        self.assertTrue(result['proceed_to_independent_evaluation'])
        for mean,c10 in [(good,strong),(strong,good)]:
            result=api.patch_development_gate(baseline,selected,mean,c10)
            self.assertFalse(result['proceed_to_independent_evaluation'])


if __name__=='__main__':unittest.main()
