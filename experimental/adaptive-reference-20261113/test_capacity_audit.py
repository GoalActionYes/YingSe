import unittest
import numpy as np
import torch
import torch.nn.functional as F
from capacity_audit import block_split,polynomial_features,ridge_fit,rgb_collision_floor,native_features,render_coefficients,post_forward


class LinearGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ['lut_L','lut_Ct','lut_Cp']:self.register_buffer(name,torch.zeros(64))
    def _mono(self,w,lo,hi):return torch.linspace(lo,hi,64,device=w.device,dtype=w.dtype)
    def forward(self,x):return ((x+1)/2).clamp(0,1)


class Sampler(torch.nn.Module):
    def __init__(self):super().__init__();self.L_weights=torch.nn.Parameter(torch.zeros(1,3,1,1));self.pre_guide=torch.nn.Identity()
    def forward(self,grid,c):
        _,_,h,w=c.shape;y,x=torch.meshgrid((2*torch.arange(h)+1)/h-1,(2*torch.arange(w)+1)/w-1,indexing='ij')
        z=(c*F.softmax(self.L_weights,1)).sum(1);coords=torch.stack([x[None],y[None],(z-.5)*2],-1)[:,None]
        return F.grid_sample(grid,coords,align_corners=False,padding_mode='border').squeeze(2)


class TinyModel(torch.nn.Module):
    def __init__(self):super().__init__();self.pre_guide=LinearGuide();self.post_guide=LinearGuide();self.sampler=Sampler()
    def apply_coeffs(self,coeff,c):
        value=coeff[:,:12]/coeff[:,12:];matrix=value[:,:9].reshape(1,3,3,*c.shape[2:]);raw=torch.einsum('boihw,bihw->bohw',matrix,self.pre_guide((c-.5)*2))+value[:,9:]
        return self.post_guide(raw.clamp(-1,1))


def lowpass(x):
    shape=x.shape;return F.avg_pool2d(x.reshape(-1,1,16,16),3,1,1,count_include_pad=False).reshape(shape)


class CapacityAuditTests(unittest.TestCase):
    def test_heldout_truth_cannot_change_fitted_coefficients(self):
        rgb=np.random.default_rng(5).integers(0,256,(16,16,3),dtype=np.uint8);x=polynomial_features(rgb,1);mask=block_split(16,16,4)
        truth=x[:,:3]*.6+.1;changed=truth.copy();changed[~mask]=1
        np.testing.assert_array_equal(ridge_fit(x,truth,mask),ridge_fit(x,changed,mask))
        self.assertLess(float(np.abs(x[~mask]@ridge_fit(x,truth,mask)-truth[~mask]).mean()),.001)

    def test_rgb_collision_l1_floor_uses_channel_medians(self):
        rgb=np.zeros((1,3,3),np.uint8);gt=np.array([[[0,0,0],[20,20,20],[40,40,40]]],np.uint8)
        result=rgb_collision_floor(rgb,gt);self.assertAlmostEqual(result['minimum_mae'],40/3);self.assertEqual(result['repeated_key_pixels'],3)

    def test_native_grid_design_and_renderer_agree_for_global_and_spatial_fields(self):
        rgb=np.random.default_rng(9).integers(0,256,(19,25,3),dtype=np.uint8);model=TinyModel().eval();api={'lowpass':lowpass}
        features=native_features(model,api,rgb);rng=np.random.default_rng(10)
        for name in ['guided_global','guided_spatial']:
            x=features[name];coef=rng.normal(0,.05,(x.shape[1],3));expected=post_forward(x@coef,features['post_tables']).reshape(rgb.shape)
            actual=render_coefficients(model,api,rgb,coef)
            self.assertLessEqual(int(np.abs(actual.astype(int)-expected.astype(int)).max()),1)

    def test_invalid_splits_and_nonfinite_fits_fail(self):
        with self.assertRaises(ValueError):block_split(2,2,32)
        with self.assertRaises(ValueError):ridge_fit(np.full((4,4),np.nan),np.zeros((4,3)),np.array([1,1,0,0],bool))

    def test_known_smooth_spatial_field_recovers_on_unfitted_blocks(self):
        rgb=np.random.default_rng(18).integers(80,176,(32,40,3),dtype=np.uint8);model=TinyModel().eval();api={'lowpass':lowpass};features=native_features(model,api,rgb)
        params=np.zeros((9,8,4,3))
        for node in range(9):
            params[node,:,:3,:]=np.eye(3)*2;params[node,:,3,:]=-1
            params[node,:,3,0]+=.2*(node%3-1);params[node,:,3,2]-=.1*(node%3-1)
        gt=render_coefficients(model,api,rgb,params.reshape(288,3));tables=features['post_tables'];domain=np.linspace(-1,1,64)
        target=np.column_stack([np.interp(gt.reshape(-1,3)[:,c]/255,tables[:,c],domain) for c in range(3)]);mask=block_split(32,40,4)
        global_fit=ridge_fit(features['guided_global'],target,mask);spatial_fit=ridge_fit(features['guided_spatial'],target,mask,regions=9)
        a=render_coefficients(model,api,rgb,global_fit).reshape(-1,3);b=render_coefficients(model,api,rgb,spatial_fit).reshape(-1,3);truth=gt.reshape(-1,3)
        global_error=float(np.abs(a[~mask].astype(float)-truth[~mask]).mean());local_error=float(np.abs(b[~mask].astype(float)-truth[~mask]).mean())
        self.assertLess(local_error,1);self.assertGreater(global_error,local_error+1)


if __name__=='__main__':unittest.main()
