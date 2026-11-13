import importlib
import importlib.util
import unittest
import torch
import torch.nn.functional as F
from pairwise_context import enable_pairwise_context,pairwise_grid,pairwise_histograms

class TinyAttention(torch.nn.Module):
    def forward(self,x,context):return context[:,:,:,None].expand_as(x)

class Resize:
    @staticmethod
    def resize(x,size,antialias=True):return F.interpolate(x,size,mode='bilinear',align_corners=False,antialias=antialias)

class TinyGrid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_projection=torch.nn.Sequential(torch.nn.AvgPool2d(4),torch.nn.Conv2d(3,128,3,padding=1),torch.nn.GELU())
        self.hist_projection=torch.nn.Sequential(torch.nn.Conv1d(6,128,1),torch.nn.AdaptiveAvgPool1d(1))
        self.attention=TinyAttention();self.fusion=torch.nn.Conv2d(256,104,1)

def ycbcr(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    return torch.cat([.2126*r+.7152*g+.0722*b,-.1146*r-.3854*g+.5*b,.5*r-.4542*g-.0458*b],1)

API={'srgb_to_ycbcr':ycbcr,'batch_histogram_1D':lambda x:x.mean((-2,-1))[:,:,None].expand(-1,-1,32),'TF':Resize,'lowpass':lambda x:x}

class AppearancePairwiseTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('appearance_pairwise'))
        return importlib.import_module('appearance_pairwise')
    def model(self):
        torch.manual_seed(20261123);m=TinyGrid();enable_pairwise_context(m);self.module().enable_appearance_context(m);return m
    def test_zero_head_preserves_pairwise_grid_and_can_learn(self):
        module=self.module();m=self.model();c=torch.rand(1,3,64,64);s=torch.rand_like(c)
        expected=pairwise_grid(m,API,c,s);actual=module.appearance_grid(m,API,c,s)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        actual.sum().backward();gradient=m.reference_adapter.layers[-1].weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()));self.assertGreater(int(torch.count_nonzero(gradient)),0)
    def test_appearance_can_read_organization_beyond_color_counts(self):
        module=self.module();m=self.model();c=torch.rand(1,3,64,64);s=torch.full_like(c,.15);s[:,:,:32]=.8
        order=torch.randperm(64*64);other=s.flatten(2)[:,:,order].reshape_as(s)
        torch.testing.assert_close(pairwise_histograms(ycbcr(s)),pairwise_histograms(ycbcr(other)),atol=0,rtol=0)
        with torch.no_grad():m.reference_adapter.layers[-1].weight.normal_(0,.1)
        a=module.appearance_grid(m,API,c,s);b=module.appearance_grid(m,API,c,s,appearance_reference=other)
        self.assertGreater(float((a-b).abs().max()),1e-6)
    def test_disabled_branch_exactly_delegates_existing_pairwise_grid(self):
        module=self.module();m=self.model();c=torch.rand(1,3,64,64);s=torch.rand_like(c)
        with torch.no_grad():m.reference_adapter.layers[-1].weight.normal_(0,.1)
        torch.testing.assert_close(module.appearance_grid(m,API,c,s,disable_appearance=True),pairwise_grid(m,API,c,s),atol=0,rtol=0)
    def test_invalid_images_modes_and_duplicate_installation_fail(self):
        module=self.module();m=self.model();c=torch.rand(1,3,64,64)
        for ref in [torch.full_like(c,float('nan')),torch.zeros(2,3,64,64),torch.zeros(1,4,64,64),torch.full_like(c,2.)]:
            with self.assertRaises(ValueError):module.appearance_grid(m,API,c,ref)
        with self.assertRaises(ValueError):module.appearance_grid(m,API,c,c,disable_appearance=1)
        with self.assertRaises(ValueError):module.appearance_grid(m,API,c,c,disable_appearance=True,appearance_reference=c)
        with self.assertRaises(ValueError):module.enable_appearance_context(m)
        with self.assertRaises(ValueError):module.enable_appearance_context(TinyGrid())

if __name__=='__main__':unittest.main()
