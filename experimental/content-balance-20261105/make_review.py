"""Build a local self-contained review from executed evidence, not a new App."""
from pathlib import Path
import sys,json,base64,io,html
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'outputs/content-balance-20261105/holdout-v3'
OUT=ROOT/'讨论/14-构图适配研究验收'

def data_image(path,style=False):
 im=Image.open(path).convert('RGB')
 if style:im=Image.fromarray(np.rint(np.clip(np.asarray(im).astype(float)*[1.04,.98,.92]+[3,1,0],0,255)).astype(np.uint8))
 im.thumbnail((420,420));b=io.BytesIO();im.save(b,format='PNG')
 return f'<img width="{im.width}" height="{im.height}" alt="已执行试验的图像对照" src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'

def main():
 OUT.mkdir(exist_ok=True)
 report=json.loads((DATA/'results.json').read_text('utf8'));figures=[];bars=[]
 names={'astronaut.png':'宇航员','grace_hopper.jpg':'正装人像','male_full_height_hands.jpg':'全身人物','cat.jpg':'非人物边界：猫'}
 for r in report['records']:
  name=r['file'];stem=Path(name).stem;label=names[name]
  cells=[]
  for text,path,style in [('目标原图',DATA/'assets'/name,False),('已知控制参考',DATA/'assets'/name,True),
                           ('全图统计对照',DATA/(stem+'-amount-0.png'),False),('参考构图适配',DATA/(stem+'-amount-0.5.png'),False)]:
   cells.append('<div><p class="ve-label">'+text+'</p>'+data_image(path,style)+'</div>')
  error=r['controlled'][1]['mae']-r['controlled'][0]['mae']
  figures.append(f'<figure><h3>{label}</h3><div class="photos">'+''.join(cells)+f'</div><figcaption>同内容控制误差变化 {error:+.4f} 个8位码值；接近零不代表整体调色误差为零。</figcaption></figure>')
  if r['occupancy']:
   v=r['occupancy']['reduction']*100
   bars.append(f'<ve-bar value="{v:.1f}" hot>{label}</ve-bar>')
 sources=json.loads((DATA/'sources.json').read_text('utf8'))
 links=' · '.join(f'<a href="{html.escape(s["url"])}">{html.escape(s["file"])}</a>' for s in sources)
 source='''<!doctype html><html lang="zh-CN" data-register="instrument"><head><title>映色：参考构图适配研究验收</title>
<link rel="stylesheet" href="kit.css" data-ve-kit>
<style>
main{max-width:1280px}.photos{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--s-3)}
.photos img{display:block;width:100%;height:300px;object-fit:contain;background:var(--surface)}
.photos p{margin-bottom:var(--s-2)}.meta,code{overflow-wrap:anywhere}
@media(max-width:760px){.photos{grid-template-columns:repeat(2,minmax(0,1fr))}.photos img{height:220px}}
</style></head><body><main><header><p class="eyebrow">映色 · 2026年11月5日 · 研究版 v3</p>
<h1>减少构图干扰，还没有证明普遍更好看</h1>
<p class="lead">新方法保持目标统计不变，只调整参考图中人物与背景对统计的贡献，最后仍生成一套全画面颜色映射。</p>
<p><span class="chip ok">新增三组受控占比通过</span> <span class="chip warn">未接入默认 App</span></p></header>
<figure><ve-bars unit="%" max="100" label="三组新照片的受控占比漂移分别减少约47%、66%、55%">'''+''.join(bars)+'''</ve-bars>
<figcaption>相对全像素未加权对照，参考人物占比从20%变到80%时的输出漂移减少；这不是画质评分。</figcaption></figure>
<section><h2>两张图只生成一套共同映射</h2><figure>
<ve-flow label="保持目标特征，只适配参考贡献；共同映射应用整个图像或视频。">
<ve-node id="target" at="1 1" sub="保持统计" key>目标图或帧</ve-node>
<ve-node id="share" at="1 2" sub="不改变目标统计">目标人物占比</ve-node>
<ve-node id="ref" at="1 3" sub="分割人物与背景">参考图</ve-node>
<ve-node id="weights" at="2 3" sub="贡献限制0.5至2倍">参考统计</ve-node>
<ve-node id="map" at="3 1" sub="复用恢复的下游公式">共同颜色映射</ve-node>
<ve-node id="output" at="4 1" sub="整张图或整段视频">输出</ve-node>
<ve-edge from="ref" to="weights" label="适配目标占比" hot/>
<ve-edge from="target" to="share" label="分割"/>
<ve-edge from="share" to="weights" label="提供人物占比"/>
<ve-edge from="target" to="map" label="提取未加权特征"/>
<ve-edge from="weights" to="map" label="生成参考特征"/>
<ve-edge from="map" to="output" label="全画面应用"/>
</ve-flow><figcaption>这是自主研究的前端统计改进，不是证明剪映使用同一构图适配方法，也不是人物、背景各套一张滤镜。</figcaption></figure></section>
<section><h2>先看是否额外破坏本来相近的构图</h2><p>左二是统一温和变换生成的控制参考。右侧两列都以80%强度输出；两者使用相同全像素估计器，未与稳定版稀疏提取路径混为同一对照。</p>'''+''.join(figures)+'''</section>
<section><h2>目前还不能据此发布默认增强</h2>
<p>18项代码测试通过，3秒视频75帧完整解码通过。恒定颜色的浮点残差问题已修复；已恢复的矩阵公式没有修改。</p>
<p>猫图仍被分割模型识别出约10%“人物”区域。因此人物蒙版不能当作可靠的人物存在判定。不同内容参考仍可能将场景本色迁移到肤色或衣物。</p>
<p>三组新增图像支持受控占比指标，但旧像素池的一个重复对照只改善约5%。两者试验构造不同，不能合并后只报好成绩。尚无多人盲评、自然风格真值、长视频和新分支音频验证。</p>
<p>下一步先做独立自然参考的匿名对照，并验证非人物回退；不继续针对本页图片调参数。稳定版入口仍在“00-从这里开始”。</p></section>
<section class="sources" aria-label="证据来源"><h2>可复核证据</h2><p>'''+links+'''</p>
<p><code>experimental/content-balance-20261105/holdout_v3_protocol.md</code> 固定了验收协议。</p>
<p><code>docs/research/content-balance-20261105/reference-composition-v3/</code> 保存结果、失败记录及素材摘要。</p>
<p>原图版权与许可按各来源项目或素材规定；本页用于本地教学研究展示，未随源码上传图像。</p></section></main></body></html>'''
 (OUT/'构图适配研究展示.src.html').write_text(source,'utf8')
 print(OUT/'构图适配研究展示.src.html')

if __name__=='__main__':main()
