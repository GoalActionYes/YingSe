"""Standalone anonymous rating page; no model key or existing AI votes exported."""
import argparse,base64,hashlib,io,json
from pathlib import Path
from PIL import Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();folder=args.directory
    manifest=json.loads((folder/'manifest.json').read_text());cards=[]
    for sheet in range(1,9):
        path=folder/f'sheet-{sheet}.png'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sheets'][path.name]:raise ValueError('Changed sheet')
        with Image.open(path) as im:
            for row in range(4):
                buf=io.BytesIO();im.crop((0,row*292,1024,(row+1)*292)).save(buf,format='PNG')
                cards.append('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
    identity=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>参考风格匿名评价</title>
<style>body{background:#202328;color:#eee;font:16px/1.7 system-ui;max-width:1120px;margin:24px auto;padding:16px}button,input{font:inherit}button{padding:8px 16px;margin:8px;background:#76d9cf;border:0;border-radius:6px}img{width:100%;background:#111}fieldset{display:inline-block;vertical-align:top;margin:8px;padding:12px}label{margin:8px;display:inline-block}input[type=text]{max-width:180px}small{color:#ccc}</style>
<h1>参考风格匿名评价</h1><p>左起为目标原图、参考图、结果A、结果B。请独立评价三个维度；不是判断内容是否相同。允许平局或无法判断。用编号或昵称填写评价者，不必填写真实姓名。</p>
<p>风格接近：冷暖、明暗、饱和度及对比关系是否接近参考。自然度：肤色、白色物体、明暗与细节是否合理。协调性：人物／主体和环境是否统一、是否出现突兀偏色。</p>
<p><label>评价者 <input id="reviewer" type="text" maxlength="40"></label><span id="progress"></span></p><h2 id="case"></h2><img id="panel" alt="匿名四列对照"><div id="votes"></div>
<button id="prev">上一组</button><button id="next">下一组</button><button id="export">导出评价JSON</button><p id="message" role="status"></p>
<small>本页离线运行，不上传素材或评价。自动保存仅在当前浏览器支持本地存储时可用；离开前请导出。未填完的数据会标记为部分评价。材料包含32组已用于研究的图片，不属于新的独立测试，不提供标准答案，也不显示算法身份。请不要查看映射文件后再评分。</small>
<script>
const cards=CARDS,manifest=MANIFEST,storage='yingse-review-'+manifest;
let index=0,state={reviewer:'',ratings:Array.from({length:32},(_,i)=>({alias:'C'+String(i+1).padStart(2,'0'),style:null,naturalness:null,coordination:null}))};
const keys=['style','naturalness','coordination'],allowed=['A','B','tie','unsure'];
try{const saved=JSON.parse(localStorage.getItem(storage));if(saved&&typeof saved.reviewer==='string'&&saved.ratings?.length===32&&saved.ratings.every((r,i)=>r.alias===state.ratings[i].alias&&keys.every(k=>r[k]===null||allowed.includes(r[k]))))state={reviewer:saved.reviewer,ratings:saved.ratings.map(r=>({alias:r.alias,style:r.style,naturalness:r.naturalness,coordination:r.coordination}))};}catch(e){}
const $=id=>document.getElementById(id);$('reviewer').value=state.reviewer;
function save(){state.reviewer=$('reviewer').value;try{localStorage.setItem(storage,JSON.stringify(state));}catch(e){$('message').textContent='浏览器未能自动保存，请使用导出按钮。';}progress();}
function progress(){const n=state.ratings.filter(r=>keys.every(k=>allowed.includes(r[k]))).length;$('progress').textContent='已完成 '+n+' / 32组';return n;}
function render(){$('case').textContent=state.ratings[index].alias+' · 第'+(index+1)+'组';$('panel').src=cards[index];$('votes').replaceChildren();
keys.forEach((key,i)=>{const box=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=['风格接近','自然度','协调性'][i];box.append(legend);
allowed.forEach((value,j)=>{const label=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name=key;input.value=value;input.checked=state.ratings[index][key]===value;input.onchange=()=>{state.ratings[index][key]=value;save();};label.append(input,document.createTextNode(['A更好','B更好','平局','无法判断'][j]));box.append(label);});$('votes').append(box);});$('prev').disabled=index===0;$('next').disabled=index===31;progress();}
$('prev').onclick=()=>{index--;render();};$('next').onclick=()=>{index++;render();};$('reviewer').oninput=save;
$('export').onclick=()=>{save();if(!state.reviewer.trim()){$('message').textContent='请填写评价者编号或昵称。';return;}const completed=progress();const result={schema_version:1,material_sha256:manifest,reviewer_kind:'self_reported',completed_cases:completed,total_cases:32,complete:completed===32,...state};const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='reference-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('message').textContent='已导出'+completed+'/32组。平局与无法判断将单独统计，不计为任一方胜出。';};render();
</script></html>'''
    html=html.replace('CARDS',json.dumps(cards)).replace('MANIFEST',json.dumps(identity))
    out=folder/'匿名评价.html';out.write_text(html,encoding='utf-8');print(json.dumps(dict(cards=len(cards),bytes=out.stat().st_size,material_sha256=identity,scope='Empty anonymous human review page, not completed human ratings')))


if __name__=='__main__':main()
