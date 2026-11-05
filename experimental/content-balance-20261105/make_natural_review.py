"""Create anonymous side-by-side natural-reference cases; not aesthetic results."""
from pathlib import Path
import sys,json,html,itertools
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from check_person_gate import fit,OUT as DATA
from check_natural import load
from person_gate import segment,fullmask
from dense_features import extract_dense
from make_review import data_image,ROOT,OUT
import evaluate as e

CASES=DATA/'natural-pairs'

def main():
 CASES.mkdir(exist_ok=True);rng=np.random.default_rng(20261105);inputs={};records=[];figures=[]
 for name in ('lena.jpg','messi5.jpg','basketball1.png','fruits.jpg'):
  path=DATA/'assets'/name;a=load(path);m,_,d=segment(a)
  inputs[name]=(a,fullmask(m,a),path,d)
 pairs=list(itertools.permutations(('lena.jpg','messi5.jpg','basketball1.png'),2))+[('lena.jpg','fruits.jpg')]
 for index,(target_name,ref_name) in enumerate(pairs,1):
  a,am,ap,ad=inputs[target_name];b,bm,bp,bd=inputs[ref_name]
  baseline=e.build_mapping(extract_dense(a,0),extract_dense(b,1));adapted,diag=fit(a,b,am,bm)
  variants={'unweighted_dense':e.apply_mapping(a,baseline,.8),'face_gated_composition':e.apply_mapping(a,adapted,.8)}
  order=['unweighted_dense','face_gated_composition']
  if rng.integers(2):order.reverse()
  case=f'case{index:02d}';paths=[]
  for label,method in zip(('A','B'),order):
   path=CASES/(case+'-'+label+'.png');Image.fromarray(variants[method]).save(path);paths.append(path)
  identical=bool(np.array_equal(variants['unweighted_dense'],variants['face_gated_composition']))
  records.append(dict(case_id=case,target=target_name,reference=ref_name,A=order[0],B=order[1],
    identical_outputs=identical,adaptation=diag,target_evidence=ad,reference_evidence=bd))
  images=''.join('<div><p class="ve-label">'+label+'</p>'+data_image(path)+'</div>' for label,path in zip(('目标原图','参考图','结果 A','结果 B'),(ap,bp,*paths)))
  options=''.join('<label><input type="radio" name="'+case+'" value="'+value+'"> '+label+'</label>' for value,label in (('A','A 更合适'),('B','B 更合适'),('equal','看不出差别'),('neither','两者都不理想')))
  figures.append('<figure><h3>对照 '+str(index)+'</h3><div class="photos">'+images+'</div><fieldset><legend>综合参考风格、人物自然程度和整体协调性</legend>'+options+'</fieldset><figcaption>两列输出使用相同80%调色强度、相同预览缩放；A/B方法名称隐藏。请选择实际观感，也可以暂时不选。</figcaption></figure>')
 (CASES/'key.json').write_text(json.dumps(dict(records=records,ratings_collected=0,
   warning='This is an operator method key, NOT human evaluation results; no automatic winner'),ensure_ascii=False,indent=2),'utf8')
 scoring=Path(__file__).with_name('review_scoring.js').read_text('utf8')
 source='''<!doctype html><html lang="zh-CN" data-register="instrument"><head><title>映色自然参考匿名对照</title><link rel="stylesheet" href="kit.css" data-ve-kit>
<style>main{max-width:1280px}.photos{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--s-3)}.photos img{width:100%;height:300px;object-fit:contain;background:var(--surface)}
fieldset{display:flex;flex-wrap:wrap;gap:var(--s-4);padding:var(--s-3);border:1px solid var(--border-bright);margin-top:var(--s-4)}
legend{font-size:var(--t-small)}label{cursor:pointer}button{padding:var(--s-3);font:inherit;cursor:pointer}#status{font-size:var(--t-small)}
@media(max-width:760px){.photos{grid-template-columns:repeat(2,minmax(0,1fr))}.photos img{height:220px}}</style></head><body><main><header>
<p class="eyebrow">映色 · 自然参考画面评议 · 2026年11月5日</p><h1>先看画面，<br>再判断是否有帮助</h1><p class="lead">比较两种研究方法的输出。输入是自然图片，参考不是从目标人工生成；没有公认的正确答案。</p>
<p>不显示算法名称，请按观感独立选择。相同图片重复用于不同配对，不能当作七组独立数据。预览不是完整App。</p></header><section>'''+''.join(figures)+'''</section>
<section><h2>导出评议，保留不同意见</h2><p>记录只在当前页面内存中生成，关闭前请导出；不会自动上传。可以导出未完成记录，空题会明确保留，不能算成赞同或持平。</p>
<button id="download" type="button">导出我的评议记录</button><p id="status" role="status">尚未导出。</p><p>当前尚未收集他人评分，页面不代表方法已经通过主观验收。</p></section>
<section class="sources" aria-label="证据来源"><h2>来源与范围</h2><p><a href="人物适用条件验证.html">查看人物适用条件验证</a> · <a href="构图适配研究展示.html">查看构图适配历史结果</a></p>
<p>图像来自OpenCV官方sample_data的lena.jpg、messi5.jpg、basketball1.png、fruits.jpg。原始输出与方法对应表保留在本机outputs/content-balance-20261105/person-gate-v4/natural-pairs；图像版权按来源规定，未上传Git。</p>
<p>比较全像素未加权对照与可见面部条件下的构图适配，不是稳定版与剪映的实机盲测。建议多名同学独立评议后再讨论；没有自然风格真值，不能把偏好换算成客观准确率。</p></section>
<script>'''+scoring+'''
document.getElementById('download').addEventListener('click',()=>{
 const ids='''+json.dumps([r['case_id'] for r in records])+''';const choices={};
 ids.forEach(id=>{const selected=document.querySelector('input[name="'+id+'"]:checked');if(selected)choices[id]=selected.value;});
 const result=buildSubmission(ids,choices);const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='yingse-natural-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.getElementById('status').textContent='已生成记录：'+result.ratings.filter(r=>r.choice!==null).length+'/'+ids.length+'题。'+(result.completed?'已填完。':'未填题仍保留为空。');
});</script></main></body></html>'''
 (OUT/'自然参考匿名对照.src.html').write_text(source,'utf8')
 print(json.dumps(dict(cases=len(records),identical_cases=[r['case_id'] for r in records if r['identical_outputs']],ratings_collected=0),indent=2))

if __name__=='__main__':main()
