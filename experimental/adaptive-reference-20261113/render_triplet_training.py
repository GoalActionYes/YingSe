"""Inspect validation gains and failures, with no reserved inputs."""
import json
from PIL import Image,ImageDraw
from independent_coco import ROOT


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';run=base/'training-v1';data=base/'cohort-v1'
    report=json.loads((run/'results.json').read_text(encoding='utf-8'));epoch=report['best_epoch']
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));rows={r['index']:r for r in cohort['groups']['validation']}
    prepared=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'));lookup={r['original_path']:data/r['prepared_path'] for r in prepared['files']}
    scores=json.loads((run/f'epoch-{epoch}/scores.json').read_text(encoding='utf-8'));current={r['index']:r['mae'] for r in report['current']['cases']}
    ranking=sorted(scores['cases'],key=lambda r:r['mae']-current[r['index']]);chosen=ranking[:2]+ranking[-3:]
    for score in chosen:
        i=score['index'];row=rows[i];panel=Image.new('RGB',(900,640),'#222222');draw=ImageDraw.Draw(panel)
        paths=[('Content',lookup[row['content']]),('Reference',lookup[row['reference']]),('GT',lookup[row['gt']]),
            ('Current',run/'current'/f'{i}.png'),('Before fine-tune',run/'before'/f'{i}.png'),('After fine-tune',run/f'epoch-{epoch}'/f'{i}.png')]
        for n,(label,path) in enumerate(paths):
            with Image.open(path) as im:
                im=im.convert('RGB');im.thumbnail((296,278));x,y=n%3*300,n//3*310;panel.paste(im,(x,y+25));draw.text((x+6,y+5),label,fill='white')
        draw.text((10,620),f'validation index {i}; current MAE {current[i]:.3f}; candidate {score["mae"]:.3f}',fill='white')
        panel.save(run/f'{i}-comparison.jpg',quality=95)
    print([r['index'] for r in chosen])


if __name__=='__main__':main()
