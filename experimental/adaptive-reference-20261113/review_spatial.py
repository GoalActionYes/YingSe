"""Fixed first-photo visual contact sheet; all numeric cases remain in results.json."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping

out=ROOT/'outputs/adaptive-reference-20261113/spatial-development'
sourcepath=ROOT/'outputs/adaptive-reference-20261113/historical-development/target-0.png'
source=np.asarray(Image.open(sourcepath))
sheet=Image.new('RGB',(1024,4*300),'#202020')
draw=ImageDraw.Draw(sheet)
for row,name in enumerate(('brightness','contrast','saturation','color')):
    refpath=out/f'0-{name}-reference.png'
    baseline=apply_mapping(source,analyze(sourcepath,refpath),1.)
    arrays=[source,np.asarray(Image.open(refpath)),baseline,np.asarray(Image.open(out/f'0-{name}-result.png'))]
    for column,(label,array) in enumerate(zip(('target','reference','global','candidate'),arrays)):
        image=Image.fromarray(array)
        image.thumbnail((256,270),Image.Resampling.BILINEAR)
        sheet.paste(image,(column*256+(256-image.width)//2,row*300+30))
        draw.text((column*256+4,row*300+6),name+' / '+label,fill='white')
sheet.save(out/'first-photo-review.jpg')
print(str(out/'first-photo-review.jpg'))
