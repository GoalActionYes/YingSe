"""Local visual record of the visible-face eligibility experiment."""
from pathlib import Path
import sys,json,html
sys.path.insert(0,str(Path(__file__).resolve().parent))
from make_review import data_image,ROOT,OUT

DATA=ROOT/'outputs/content-balance-20261105/person-gate-v4'

def locate(name):
 for path in (DATA/'assets'/name,ROOT/'outputs/content-balance-20261105/holdout-v3/assets'/name,
              ROOT/'outputs/content-balance-20261105/natural-check/assets'/name,
              ROOT/'experimental/skin-protection-20261019/reference_color_app/demo'/name):
  if path.exists():return path
 raise FileNotFoundError(name)

def main():
 OUT.mkdir(exist_ok=True);report=json.loads((DATA/'results.json').read_text('utf8'))
 records=report['records'];people=sum(r['expected_contains_person'] for r in records)
 negatives=len(records)-people;accepted=sum(r['new']['person_detected'] for r in records)
 assert people==accepted and not report['false_accepts'] and not report['person_images_that_fell_back']
 statuses=[];details=[]
 labels={'cat.jpg':'旧猫图：修复已知误触发','messi5.jpg':'新增小人物：保留适配',
         'basketball1.png':'新增多人运动：保留适配','fruits.jpg':'新增非人物：回退'}
 for r in records:
  name=r['file'];stem=Path(name).stem;allowed=r['new']['person_detected']
  status='允许适配' if allowed else '回退全图'
  split='新增验证' if r['split']=='new-fixed-validation' else '已观察回归'
  statuses.append('<div class="sample">'+data_image(locate(name))+'<p><b>'+html.escape(name)+'</b></p><p><span class="chip '+('ok' if allowed else 'warn')+'">'+status+'</span> '+split+'</p></div>')
  if name in labels:
   items=[('原图',locate(name)),('旧人物蒙版',DATA/'masks'/(stem+'-raw.png')),('增加适用条件后',DATA/'masks'/(stem+'-gated.png'))]
   columns=''.join('<div><p class="ve-label">'+label+'</p>'+data_image(path)+'</div>' for label,path in items)
   face=r['new']['face_evidence']
   details.append('<figure><h3>'+labels[name]+'</h3><div class="comparison">'+columns+'</div><figcaption>白色表示统计时的人物区域；最大高置信面部连通区域为'+str(face['largest_face_component'])+'像素，本图最低要求为'+str(face['minimum_face_component'])+'像素。'+('通过后蒙版与旧输出逐值相同。' if allowed else '条件不足时蒙版归零，不再执行人物贡献适配。')+'</figcaption></figure>')
 source='''<!doctype html><html lang="zh-CN" data-register="instrument"><head><title>映色：人物适用条件验证</title>
<link rel="stylesheet" href="kit.css" data-ve-kit><style>
main{max-width:1280px}.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--s-3)}
.comparison img{width:100%;height:290px;object-fit:contain;background:var(--surface)}
.samples{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--s-4)}
.sample img{width:100%;height:160px;object-fit:contain;background:var(--surface)}
.sample p{font-size:var(--t-small);overflow-wrap:anywhere}.sample{border-bottom:1px solid var(--border);padding-bottom:var(--s-3)}
@media(max-width:760px){.samples{grid-template-columns:repeat(2,minmax(0,1fr))}.comparison{grid-template-columns:1fr}.comparison img{height:240px}}
</style></head><body><main><header><p class="eyebrow">映色 · 2026年11月5日 · 研究版 v4</p>
<h1>先查面部证据，<br>再启用人物适配</h1><p class="lead">修复旧猫图误触发：条件不足时回到普通全图统计，不强行猜人物。调色映射公式保持原样。</p>
<p><span class="chip ok">本组回退验证通过</span> <span class="chip warn">不是通用人物检测</span></p></header>
<figure><ve-bars unit="张" max="13" label="本组9张人物保留，4张非人物回退"><ve-bar value="'''+str(people)+'''" hot>人物保留</ve-bar><ve-bar value="'''+str(negatives)+'''">非人物回退</ve-bar></ve-bars>
<figcaption>已观察回归与六张新增验证合计13张；固定阈值后未因新增结果调参。此小样本不能解释为真实场景准确率。</figcaption></figure>
<section><h2>条件改变是否允许适配，不改变通过后的蒙版</h2>'''+''.join(details)+'''</section>
<section><h2>全部样例保留，避免只看成功截图</h2><figure><div class="samples">'''+''.join(statuses)+'''</div><figcaption>9张人物没有在本组被拒绝；4张非人物均回退。新增六张中人物与非人物各三张。</figcaption></figure></section>
<section><h2>回退已核对，画质提升仍待验证</h2><p>25项代码测试通过。四张被拒绝图像分别作为目标、参考，共八次检查的LUT与输出均与未加权全像素对照逐值一致。通过条件的九张图像，蒙版与旧方法逐值一致。视频完成3秒75帧整片解码。</p>
<p>回退对照仍是研究用全像素统计，不是稳定版稀疏提取。可见面部证据来自同一个分割模型，不是独立人物检测器；高置信度不等于校准后的可靠概率。</p>
<p>背影、遮脸、暗光和极小面部可能无法通过，回退不会停止调色。仍可能有动物或物体被误认面部；这里没有验证全部物种、肤色、遮挡和光照。</p>
<p>本轮没有新增界面控件、模型或默认App行为。多人盲评、独立自然风格参考、长片和音频仍待验收。</p>
<p><a href="构图适配研究展示.html">查看上一轮v3的构图适配与画面质量限制</a></p></section>
<section><h2>下一步直接看实际调色画面</h2><p><a href="自然参考匿名对照.html">打开七组自然参考匿名对照并导出评议记录</a>。这里只准备了对照，尚未收集他人评分，不代表主观验收通过。</p></section>
<section class="sources" aria-label="证据来源"><h2>证据与复现</h2>
<p>'''+ ' · '.join('<a href="'+html.escape(s['url'])+'">'+html.escape(s['file'])+'</a>' for s in json.loads((DATA/'sources.json').read_text('utf8')))+'''</p>
<p><code>experimental/content-balance-20261105/person_gate_protocol.md</code> 固定了新增素材运行前的协议。</p>
<p><code>docs/research/content-balance-20261105/visible-face-v4/</code> 保存原始类别诊断、完整结果、来源摘要和测试记录。</p>
<p>公开测试图像仅嵌入本地教学展示，不随源码上传；原图版权和许可按来源规定。</p></section></main></body></html>'''
 (OUT/'人物适用条件验证.src.html').write_text(source,'utf8')
 print(OUT/'人物适用条件验证.src.html')

if __name__=='__main__':main()
