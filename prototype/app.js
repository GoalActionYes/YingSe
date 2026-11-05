'use strict';
const PALETTE='cinematic-v2';
const $=id=>document.getElementById(id);
const desktopMode=new URLSearchParams(location.search).get('desktop');
if(desktopMode==='1'||desktopMode==='portable'){
  document.querySelector('.version').textContent=desktopMode==='portable'?'V0.2.2 / 小组体验版':'V0.2.1 / 桌面演示版';
  document.title=desktopMode==='portable'?'映色 · 小组体验版':'映色 · 桌面演示版';
}
const labels=['原色','黑金','赛博朋克','青橙','复古胶片','日系清新','冷峻电影','暖阳','黑白电影'];
const controlSpecs=[['exposure','曝光',-3,3,.1,'EV'],['contrast','对比度',-100,100,1,''],['saturation','饱和度',-100,100,1,''],['temperature','色温',-100,100,1,'']];
let source='demo-source',reference='demo-reference',weights=[1,0,0,0,0,0,0,0,0],isVideo=false,job=null,engineOK=false,stateVersion=0,busyCount=0,timer;
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error)}
function adjustments(){return Object.fromEntries(controlSpecs.map(([key])=>[key,Number($(key).value)]))}
function resetAdjustments(){for(const [key] of controlSpecs)$(key).value=0}
function body(){return {palette_version:PALETTE,source,reference,weights:weights.slice(),strength:Number($('strength').value)/100,adjustments:adjustments()}}
async function api(path,data){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok){const e=await r.json();throw Error(e.error||'操作失败')}return r}
async function guarded(fn){try{busyCount++;$('recommend').disabled=true;await fn()}catch(e){status(e.message,true)}finally{busyCount--;$('recommend').disabled=busyCount>0}}
function sync(){
  const sum=weights.reduce((a,b)=>a+b,0),active=weights.map((v,i)=>v>1e-8?i:-1).filter(i=>i>=0),isOriginal=!sum||active.every(i=>i===0);
  for(let i=1;i<9;i++){$('w'+i).value=weights[i]*100;$('v'+i).textContent=sum?((weights[i]/sum)*100).toFixed(1)+'%':'0%'}
  document.querySelectorAll('.style-card').forEach((el)=>el.setAttribute('aria-pressed',String(active.length===1&&active[0]===Number(el.dataset.index))));
  $('original').setAttribute('aria-pressed',String(isOriginal));
  const name=isOriginal?'原色':active.length===1?labels[active[0]]:'混合风格';
  const changed=Object.values(adjustments()).some(v=>v!==0);
  $('activeStyle').textContent=`当前效果：${name}${isOriginal?'':' · '+$('strength').value+'%'}${changed?' ＋ 基础调整':''}`;
  $('strengthValue').textContent=$('strength').value+'%';
  for(const [key,,,,,unit] of controlSpecs){const v=Number($(key).value);$('value-'+key).textContent=(v>0?'+':'')+(key==='exposure'?v.toFixed(1):v)+(unit?' '+unit:'')}
  $('export').disabled=!isVideo||!engineOK||!!job;
}
async function preview(){const version=stateVersion;const data=await(await api('/api/preview',body())).json();if(version!==stateVersion)return;$('before').src=data.before;$('after').src=data.after}
function changed(){stateVersion++;sync();if(!job)status($('activeStyle').textContent+'；正在更新预览。');clearTimeout(timer);timer=setTimeout(()=>guarded(async()=>{await preview();if(!job)status($('activeStyle').textContent+'；预览已更新。')}),180)}
function selectStyle(index){weights=labels.map((_,i)=>i===index?1:0);$('strength').value=100;changed();status(index?`已选择${labels[index]}，强度100%。可独立调整滤镜强度和基础参数。`:'已关闭滤镜；基础调整保留。')}
async function gallery(){const key=source;const data=await(await api('/api/styles',body())).json();if(key!==source)return;
  $('styles').replaceChildren();
  data.styles.forEach((style,i)=>{if(!i)return;const button=document.createElement('button');button.className='style-card';button.dataset.index=i;button.setAttribute('aria-label','应用'+style.name+'滤镜');
    const img=document.createElement('img');img.src=style.image;img.alt=style.name+'效果预览';
    const title=document.createElement('strong');title.textContent=style.name;const desc=document.createElement('small');desc.textContent=style.description;
    button.append(img,title,desc);button.onclick=()=>selectStyle(i);$('styles').append(button)});sync();
}
for(const [key,name,min,max,step] of controlSpecs){
  const label=document.createElement('label');label.className='range';label.innerHTML=`<div><span>${name}</span><output id="value-${key}">0</output></div><input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="0" aria-label="${name}">`;
  $('adjustments').append(label);$(key).oninput=changed;
}
for(let i=1;i<9;i++){const label=document.createElement('label');label.className='range';label.innerHTML=`<div><span>${labels[i]}</span><output id="v${i}">0%</output></div><input id="w${i}" type="range" min="0" max="100" step="0.1" value="0" aria-label="${labels[i]}混合配比">`;$('sliders').append(label);$('w'+i).oninput=()=>{weights[0]=0;weights[i]=Number($('w'+i).value)/100;changed()}}
$('strength').oninput=changed;$('original').onclick=()=>selectStyle(0);
$('resetAdjustments').onclick=()=>{resetAdjustments();changed();status('基础调整已归零，所选风格保持不变。')};
$('resetAll').onclick=()=>{weights=[1,0,0,0,0,0,0,0,0];$('strength').value=100;resetAdjustments();changed();$('metrics').textContent='全部参数已恢复原色。';status('滤镜和基础调整均已重置。')};
$('recommend').onclick=()=>guarded(async()=>{const version=stateVersion;status('正在搜索风格配比…');const d=await(await api('/api/recommend',body())).json();if(version!==stateVersion){status('参数已改变，已忽略旧的推荐结果。');return}weights=d.weights;$('strength').value=Math.round(d.strength*100);resetAdjustments();stateVersion++;sync();await preview();$('metrics').textContent=`统计拟合损失：${d.metrics.rgb_moment_loss_before.toFixed(6)} → ${d.metrics.rgb_moment_loss_after.toFixed(6)}。仅代表颜色统计接近程度，不是画质评分。`;status('推荐完成；原色比例已换算为滤镜强度，基础调整已归零。')});
async function upload(file){if(file.size>300*1024*1024)throw Error('文件超过300MB，请选择较短素材');const r=await fetch('/api/upload?name='+encodeURIComponent(file.name),{method:'POST',body:file});const d=await r.json();if(!r.ok)throw Error(d.error);return d}
async function sourceChanged(){stateVersion++;sync();$('metrics').textContent='当前效果沿用已有参数。可重新选择滤镜或根据参考图推荐。';await Promise.all([preview(),gallery()])}
$('sourceFile').onchange=()=>guarded(async()=>{const file=$('sourceFile').files[0];if(!file)return;status('正在载入素材并生成风格预览…');const d=await upload(file);source=d.id;isVideo=/\.(mp4|mov|mkv|webm|avi)$/i.test(file.name);$('sourceName').textContent=file.name;await sourceChanged();status('素材已载入，风格缩略图已更新。')});
$('referenceFile').onchange=()=>guarded(async()=>{const file=$('referenceFile').files[0];if(!file)return;const d=await upload(file);reference=d.id;stateVersion++;const old=$('referenceImage').src;if(old.startsWith('blob:'))URL.revokeObjectURL(old);$('referenceImage').src=URL.createObjectURL(file);$('referenceName').textContent=file.name;status('参考图已更新，点击推荐风格。')});
function selectDemo(video){return guarded(async()=>{source=video?'demo-video':'demo-source';isVideo=video;$('sourceName').textContent=video?'内置6秒合成视频（含测试音）':'内置合成色卡';await sourceChanged();status('示例已载入。合成素材仅用于检查流程。')})}
$('demoImage').onclick=()=>selectDemo(false);$('demoVideo').onclick=()=>selectDemo(true);
async function download(path,name){const r=await api(path,body());const url=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)}
$('cube').onclick=()=>guarded(()=>download('/api/cube','style_v2.cube'));$('image').onclick=()=>guarded(()=>download('/api/image','style_preview.png'));
async function poll(id){try{const r=await fetch('/api/job/'+id);const d=await r.json();if(!r.ok)throw Error(d.error);if(d.status==='running'){status('正在导出，已处理时间：'+d.processed_time);setTimeout(()=>poll(id),800);return}job=null;$('cancel').hidden=true;sync();if(d.status==='done'){status(`导出完成，用时${d.elapsed_seconds}秒。结果使用点击导出时的参数。`);$('downloads').replaceChildren();const label=document.createElement('span');label.textContent='最近一次导出：';$('downloads').append(label);for(const [label,url]of[['MP4',d.url],['参数JSON',d.report],['LUT',d.cube]]){const a=document.createElement('a');a.className='link';a.href=url;a.download='';a.textContent=label;$('downloads').append(a)}$('resultVideo').src=d.url;$('resultVideo').style.display='block'}else status(d.message||'导出失败',d.status==='error')}catch(e){job=null;$('cancel').hidden=true;sync();status(e.message,true)}}
$('export').onclick=()=>guarded(async()=>{const d=await(await api('/api/export',body())).json();job=d.id;sync();$('cancel').hidden=false;status('正在启动编码…');poll(job)});
$('cancel').onclick=()=>guarded(async()=>{if(job)await api('/api/cancel',{id:job})});
guarded(async()=>{const info=await(await fetch('/api/info')).json();if(info.palette_version!==PALETTE)throw Error('服务仍是旧版本，请重新启动演示服务。');engineOK=!!info.ffmpeg;$('engine').textContent=engineOK?'视频引擎已就绪 · FFmpeg':'未找到FFmpeg，仍可预览图片和导出LUT';$('demoVideo').disabled=!info.demo_video||!engineOK;sync();await Promise.all([preview(),gallery()]);status('已就绪。点击风格缩略图即可完整应用滤镜。')});
