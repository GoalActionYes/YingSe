"""Research-only Torch port of chiutaiyin/PCA-Knowledge-Distillation.

Architecture adapted from the MIT-licensed author VGG student implementation,
commit 658d4a247b1aed87b9b3f19d8f8ae555e29fe31b. See PCA-KD-LICENSE.md.
This is neither recovered Jianying code nor an RGB LUT model.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from independent_coco import ROOT,digest


def _valid(x):
    return (isinstance(x,torch.Tensor) and x.ndim==4 and x.shape[0]==1
            and min(x.shape[1:])>0 and x.dtype==torch.float32
            and bool(torch.isfinite(x).all()))


def high_frequency(x):
    if not _valid(x):raise ValueError('Expected finite batch-one float32 NCHW features')
    # SYMMETRIC padding of width one includes the edge pixel, unlike REFLECT.
    return x-F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)


def zca(content,style):
    if not _valid(content) or not _valid(style) or content.shape[:2]!=style.shape[:2] or content.device!=style.device:
        raise ValueError('ZCA requires finite float32 batch-one features with matching channels/device')
    c,s=content.flatten(2),style.flatten(2)
    cm,sm=c.mean(-1,keepdim=True),s.mean(-1,keepdim=True)
    c,s=c-cm,s-sm
    eye=torch.eye(c.shape[1],device=c.device,dtype=c.dtype)[None]
    def root(centered,inverse=False):
        cov=centered@centered.transpose(1,2)/centered.shape[-1]+eye
        u,values,_=torch.linalg.svd(cov)
        powers=values.rsqrt() if inverse else values.sqrt()
        powers=torch.where(values>1e-5,powers,torch.zeros_like(powers))
        return (u*powers[:,None,:])@u.transpose(1,2)
    result=root(s)@root(c,True)@c+sm
    return result.reshape_as(content)


class PCAKD(nn.Module):
    # (input channels, output channels, kernel size), in author block order.
    ENCODER=(((3,3,1),(3,10,3)),((10,10,3),(10,20,3)),
             ((20,20,3),(20,58,3)),((58,58,3),(58,58,3),(58,58,3),(58,64,3)))
    DECODER=(((13,3,3),),((20,10,3),(10,10,3)),
             ((58,20,3),(20,20,3)),((64,58,3),(58,58,3),(58,58,3),(58,58,3)))

    def __init__(self,assets=None,device='cpu'):
        super().__init__()
        assets=Path(assets or ROOT/'.tools/PCA-KD-converted')
        manifest=json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
        if manifest['source']['revision']!='658d4a247b1aed87b9b3f19d8f8ae555e29fe31b' or not manifest['all_checkpoint_objects_restored']:
            raise ValueError('Unverified PCA-KD source/restoration')
        if digest(assets/'weights.npz')!=manifest['weights_sha256']:raise ValueError('Converted weights changed')
        state={}
        with np.load(assets/'weights.npz',allow_pickle=False) as data:
            expected=set()
            for kind,specification in [('encoder',self.ENCODER),('decoder',self.DECODER)]:
                blocks=nn.ModuleList()
                for b,spec in enumerate(specification):
                    layers=nn.ModuleList()
                    for n,(cin,cout,k) in enumerate(spec):
                        key=f'{kind}_{b}_{n}';expected.update((key+'_kernel',key+'_bias'))
                        kernel,bias=data[key+'_kernel'],data[key+'_bias']
                        if (kernel.shape!=(k,k,cin,cout) or bias.shape!=(cout,) or
                            kernel.dtype!=np.float32 or bias.dtype!=np.float32 or
                            not np.isfinite(kernel).all() or not np.isfinite(bias).all()):
                            raise ValueError('Invalid converted convolution: '+key)
                        layers.append(nn.Conv2d(cin,cout,k))
                        prefix=f'{kind}.{b}.{n}'
                        state[prefix+'.weight']=torch.from_numpy(kernel.transpose(3,2,0,1).copy())
                        state[prefix+'.bias']=torch.from_numpy(bias.copy())
                    blocks.append(layers)
                setattr(self,kind,blocks)
            if set(data.files)!=expected:raise ValueError('Unexpected converted arrays')
        self.load_state_dict(state,strict=True)
        self.eval().to(device)
        self.provenance=manifest

    @staticmethod
    def conv(layer,x,relu=True):
        if layer.kernel_size==(3,3):x=F.pad(x,(1,1,1,1),mode='reflect')
        x=layer(x)
        return F.relu(x) if relu else x

    def encode_block(self,block,x):
        skip=high_frequency(x);layers=self.encoder[block]
        if block==0:
            x=self.conv(layers[0],x,False)
            x=self.conv(layers[1],x)
        else:
            for layer in layers[:-1]:x=self.conv(layer,x)
            x=F.max_pool2d(x,2)
            x=self.conv(layers[-1],x)
        return x,skip

    def decode_block(self,block,x,skip):
        layers=self.decoder[block]
        if block==0:return self.conv(layers[0],torch.cat((x,skip),1),False)
        x=self.conv(layers[0],x)
        x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        for layer in layers[1:-1]:x=self.conv(layer,x)
        if x.shape!=skip.shape:raise ValueError('Decoder/skip dimensions differ; use multiples of eight')
        return self.conv(layers[-1],x+skip)

    def encode(self,x):
        if not _valid(x) or x.shape[1]!=3 or min(x.shape[2:])<16 or any(n%8 for n in x.shape[2:]):
            raise ValueError('Expected batch-one RGB float32, spatial dimensions >=16 and divisible by eight')
        features=[];skips=[]
        for b in range(4):
            x,skip=self.encode_block(b,x);features.append(x);skips.append(skip)
        return features,skips

    @torch.inference_mode()
    def forward(self,content,style):
        features,skips=self.encode(content);styles,_=self.encode(style)
        x=features[-1]
        for b in reversed(range(4)):x=self.decode_block(b,zca(x,styles[b]),skips[b])
        return x.clamp(0,1)
