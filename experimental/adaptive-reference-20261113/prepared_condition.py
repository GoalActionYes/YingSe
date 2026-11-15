"""Fixed-input condition fitting cache; author parameter-dependent operations retained."""
import torch
import torch.nn.functional as F
from training_checks import finite_output


class PreparedCondition:
    def __init__(self,model,api,content,reference):
        if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Backbone must be frozen eval')
        if content.ndim!=4 or content.shape[1]!=3 or content.shape[0]!=reference.shape[0] or not torch.isfinite(content).all() or not torch.isfinite(reference).all():raise ValueError('Invalid fixed RGB inputs')
        self.model,self.api,self.content=model,api,content.detach();b,_,h,w=content.shape;self.shape=content.shape
        with torch.no_grad():
            hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](content));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](reference))
            self.img=model.img_projection(api['TF'].resize(content,(64,64),antialias=True)).detach();self.context=model.hist_projection(torch.cat([hc,hs],1)).detach()
            yy=(2*torch.arange(h,device=content.device).float()+1)/h-1;xx=(2*torch.arange(w,device=content.device).float()+1)/w-1;y,x=torch.meshgrid(yy,xx,indexing='ij')
            x=x.view(1,h,w).expand(b,-1,-1);y=y.view(1,h,w).expand(b,-1,-1)
            z=(model.sampler.pre_guide(content)*F.softmax(model.sampler.L_weights,dim=1)).sum(1)
            self.coords=torch.stack([x,y,(z-.5)*2],dim=-1).view(b,1,h,w,3).clip(-1,1).detach()
            self.im=model.pre_guide((content-.5)*2).flatten(2).permute(0,2,1)[...,None].reshape(b*h*w,3,1).detach()

    def __call__(self,delta):
        finite_output(delta,(self.shape[0],128));m=self.model;b,_,h,w=self.shape
        context=self.context+delta[:,:,None]
        raw=m.fusion(torch.cat([self.img,m.attention(self.img,context)],1)).reshape(-1,13,8,16,16);alpha=F.softplus(raw[:,-1:])+1e-3
        grid=self.api['lowpass'](torch.cat([raw[:,:-1]*alpha,alpha],1))
        coeff=F.grid_sample(grid,self.coords,mode='bilinear',align_corners=False,padding_mode='border').squeeze(2)
        coeff=coeff[:,:-1]/coeff[:,-1:];coeff=coeff.flatten(2).permute(0,2,1)
        mats=coeff[...,:9].reshape(b*h*w,3,3);bias=coeff[...,9:].reshape(b*h*w,3,1)
        output=torch.bmm(mats,self.im)+bias
        output=m.post_guide(output.clamp(-1,1).reshape(b,h,w,3).permute(0,3,1,2))
        return finite_output(output,self.shape),finite_output(grid,(b,13,8,16,16))
