"""Optional official Torchvision person segmentation, evaluated against human masks."""
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image
from canonical_model import ROOT,validate_rgb

CHECKPOINT_SHA='fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70'


class PersonSegmenter:
    def __init__(self,device=None):
        import torch
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large,DeepLabV3_MobileNet_V3_Large_Weights
        self.torch=torch;self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        directory=ROOT/'.tools/torch-cache/segmentation';directory.mkdir(parents=True,exist_ok=True)
        path=directory/Path(weights.url).name
        if not path.is_file():torch.hub.download_url_to_file(weights.url,str(path),hash_prefix='fc3c493d',progress=False)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=CHECKPOINT_SHA:raise ValueError('Official segmentation checkpoint checksum mismatch')
        self.model=deeplabv3_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=21,aux_loss=True)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.model.to(self.device).eval();self.transform=weights.transforms()
        self.label=weights.meta['categories'].index('person')
        self.provenance=dict(model='DeepLabV3 MobileNetV3 Large COCO_WITH_VOC_LABELS_V1',
                             official_url=weights.url,sha256=digest,person_label=self.label,
                             torchvision_training='COCO subset with VOC categories; evaluated on project-new val content')

    def predict(self,rgb):
        validate_rgb(rgb);torch=self.torch
        start=time.perf_counter()
        image=self.transform(Image.fromarray(rgb))[None].to(self.device)
        with torch.inference_mode():
            logits=self.model(image)['out']
            probability=torch.softmax(logits,dim=1)[:,self.label:self.label+1]
            mask=torch.nn.functional.interpolate(probability,size=rgb.shape[:2],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        if not np.isfinite(mask).all():raise ValueError('Invalid automatic person probability')
        return mask,dict(seconds=time.perf_counter()-start,fraction=float((mask>.5).mean()))
