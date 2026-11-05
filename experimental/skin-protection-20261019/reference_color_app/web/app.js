'use strict';
const $ = id => document.getElementById(id);
const state = {target:null, reference:null, mapping:null, generation:0, analyzing:false, uploading:0, job:null, exportStale:false, uploadVersion:{target:0,reference:0}, previewRunning:false, previewNext:null, previewVersion:0, saveTimer:null, saveVersion:0, saveNext:null, saveRunning:false};
const cubeProtectionMessage='CUBE不包含人物区域保护，请关闭保护后导出';
function skinSelected(){return $('skin-enabled').checked===true;}
function cubeProtectionBlocked(){return skinSelected() && $('format').value==='cube';}
const status = (message, type='') => { $('status').textContent=message; $('status').className='status '+type; };
async function api(path, body, options={}) {
  let response;
  try { response=await fetch(path, body===undefined ? options : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),...options}); }
  catch (_) { throw Error('无法连接本地服务。请确认应用仍在运行，再重试。'); }
  let data;
  try { data=await response.json(); } catch (_) { throw Error('本地服务返回了无效响应，请重试。'); }
  if (!response.ok) throw Error(data.error || '处理失败，请重试。');
  return data;
}
function image(id, value) { $(id).src=value; $(id).hidden=false; }
function clearImage(id) { $(id).hidden=true; $(id).removeAttribute('src'); }
function effectiveStrength(){return $('clone-enabled').checked ? Number($('strength').value)/100 : 0;}
function syncStrength(){ $('strength-number').value=$('strength').value; }
function setView(view){
  $('comparison').dataset.view=view;
  for(const name of ['result','original','compare']){
    $('view-'+name).className='view-button'+(name===view?' active':'');
    $('view-'+name).setAttribute('aria-pressed',String(name===view));
  }
}
for(const view of ['result','original','compare']) $('view-'+view).addEventListener('click',()=>setView(view));
$('top-export').addEventListener('click',()=>{$('export-area').scrollIntoView({behavior:'smooth',block:'nearest'});$('export-area').focus({preventScroll:true});});
function setButtons() {
  const ready=!!state.mapping, waiting=state.analyzing || state.uploading>0;
  $('analyze').disabled=!state.target || !state.reference || waiting;
  $('analyze').textContent=state.analyzing?'正在分析…':'重新分析';
  $('strength').disabled=!ready || !$('clone-enabled').checked; $('strength-number').disabled=$('strength').disabled; $('reset').disabled=$('strength').disabled;
  $('export').disabled=!ready || !!state.job || waiting || cubeProtectionBlocked();
  $('top-export').disabled=!ready || waiting;
  $('demo').disabled=waiting;
  $('video-demo').disabled=waiting;
  $('format').querySelector('[value="png"]').disabled=state.target?.kind==='video';
  $('format').querySelector('[value="mp4"]').disabled=state.target?.kind!=='video';
  const note=$('format').value==='mp4'?'原尺寸视频（奇数边补齐偶数）；H.264 / AAC 有损编码，保留首条可用音轨。':$('format').value==='cube'?'.cube 包含当前启用状态和强度，供支持 3D LUT 的软件使用。':'PNG 保留原分辨率；预览显示会缩放。';
  const protectionNote=cubeProtectionBlocked()?cubeProtectionMessage:skinSelected() && state.target?.kind==='video'?'人物肤色保护视频统一导出为30fps，逐帧识别，处理可能较慢。':'';
  $('export-note').textContent=note+(state.target?.kind!=='video'?' 导入视频后可导出 MP4。':'')+(state.target?.metadata?.precision_conversion?' 输入为10位 SDR，先转为8位调色，MP4输出为8位。':'')+(protectionNote?' '+protectionNote:'');
  $('export-note').className='export-note'+(cubeProtectionBlocked()?' blocked':'');
  $('skin-note').textContent=!skinSelected()?'开启后识别目标画面的脸部与身体皮肤，减少调色偏色。默认关闭；衣物和背景照常调色。':effectiveStrength()===0?'已勾选保护；色彩克隆关闭或强度为0时，使用原色，不运行皮肤识别。':state.target?.kind==='video'?'分析帧预览会识别皮肤；导出时逐帧识别，统一30fps，处理可能较慢。':'已勾选保护；识别目标原图中的脸部与身体皮肤。请在预览完成后检查结果。';
}
function invalidate(message='素材已更改，等待分析。') {
  ++state.generation; state.mapping=null; state.analyzing=false; state.previewNext=null;
  clearTimeout(state.saveTimer); state.saveTimer=null; state.saveNext=null; ++state.saveVersion;
  clearImage('after'); $('after-placeholder').hidden=false;
  if(!$('before').hidden){image('after',$('before').src);$('after-placeholder').hidden=true;}
  $('diagnostics').hidden=true; invalidateExport(); $('preview-state').textContent='待生成映射';
  $('save-state').textContent='当前素材尚未分析'; status(message); setButtons();
}
function invalidateExport() {
  $('result-links').hidden=true;
  if(state.job) {
    state.exportStale=true;
    $('job-status').textContent='先前设置的导出仍在进行，可继续等待或取消。';
  } else {
    $('job-area').hidden=true; $('job-status').textContent=''; $('progress').style.width='0%'; $('cancel').hidden=true;
  }
}
function showMedia(which, record) {
  state[which]=record;
  $(which+'-name').textContent=record.name; $(which+'-name').title=record.name;
  const meta=record.metadata;
  $(which+'-meta').textContent=`${record.kind==='video'?'视频':'图片'} · ${meta.width} × ${meta.height}${record.kind==='video'?' · '+meta.duration_seconds.toFixed(2)+' 秒':''}${meta.precision_conversion?' · 10位 SDR → 8位处理':''}`;
  if(which==='target') {
    $('frame-control').hidden=record.kind!=='video'; $('frame-time').value=0;
    if(record.kind==='video') $('frame-time').max=Math.max(0,meta.duration_seconds-.001);
    $('format').value=record.kind==='video'?'mp4':'png';
  }
  setButtons();
}
function actualTime(metadata) {
  if(state.target?.kind==='video') $('actual-time').textContent=`实际帧时间：${metadata.actual_frame_pts_seconds?.toFixed(3) ?? '—'} 秒`;
}
async function showFrame(which) {
  const record=state[which]; if(!record) return;
  const time=which==='target'?Number($('frame-time').value):0;
  const result=await api('/api/frame',{media_id:record.id,time_seconds:time});
  if(state[which]?.id!==record.id || (which==='target' && time!==Number($('frame-time').value))) return;
  image(which+'-thumb',result.image);
  if(which==='target') { image('before',result.image); $('before-placeholder').hidden=true; actualTime(result.metadata); if(!state.mapping){image('after',result.image);$('after-placeholder').hidden=true;} }
}
async function upload(which,file) {
  if(!file) return;
  const own=++state.uploadVersion[which]; invalidate('正在导入并检查素材…');
  state[which]=null; clearImage(which+'-thumb');
  $(which+'-name').textContent=file.name; $(which+'-meta').textContent='正在检查素材…';
  if(which==='target') {clearImage('before');clearImage('after');$('before-placeholder').hidden=false;$('after-placeholder').hidden=false;$('frame-control').hidden=true;}
  ++state.uploading; setButtons();
  try {
    if(file.size>512*1024*1024 || file.size===0) throw Error('请选择非空且不超过 512 MB 的文件。');
    const record=await api('/api/media',undefined,{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(file.name)},body:file});
    if(own!==state.uploadVersion[which]) return;
    if(which==='reference' && record.kind!=='image') throw Error('参考素材必须是图片。');
    showMedia(which,record); await showFrame(which);
    if(own===state.uploadVersion[which]) status(state.target&&state.reference?'素材已就绪，即将自动分析。':'已导入，请继续选择另一份素材。');
  } catch(error) { if(own===state.uploadVersion[which]) {state[which]=null;$(which+'-name').textContent='导入失败 · 点击重新选择';$(which+'-meta').textContent='请选择支持的素材后重试';status(error.message,'error');} }
  finally {--state.uploading;setButtons();autoAnalyze();}
}
$('target-file').addEventListener('change',event=>{const file=event.target.files[0];event.target.value='';return upload('target',file);});
$('reference-file').addEventListener('change',event=>{const file=event.target.files[0];event.target.value='';return upload('reference',file);});
$('frame-time').addEventListener('input',()=>{invalidate('分析时间已更改，点击“应用此帧”更新颜色映射。');clearImage('before');clearImage('after');$('before-placeholder').hidden=false;$('after-placeholder').hidden=false;clearImage('target-thumb');$('actual-time').textContent='等待解码所选帧';});
$('frame-load').addEventListener('click',async()=>{if(!state.target)return;const gen=state.generation;$('frame-load').disabled=true;status('正在解码所选画面…','busy');try{await showFrame('target');if(gen===state.generation){status('画面已更新，即将自动分析。');autoAnalyze();}}catch(error){if(gen===state.generation)status(error.message,'error');}finally{$('frame-load').disabled=false;}});
function autoAnalyze(){if(state.target && state.reference && state.uploading===0 && !state.analyzing && !state.mapping)return analyze();}
async function analyze(){
  if(!state.target || !state.reference || state.uploading>0 || state.analyzing)return;
  invalidate('正在分析两份素材并生成颜色映射，请稍候…');
  const gen=state.generation, target=state.target, reference=state.reference;
  state.analyzing=true;setButtons();status('正在分析两份素材并生成颜色映射，请稍候…','busy');$('preview-state').textContent='分析中';
  try {
    const result=await api('/api/analyze',{target_id:target.id,reference_id:reference.id,time_seconds:Number($('frame-time').value)});
    if(gen!==state.generation)return;
    state.mapping=result.mapping_id;actualTime(result.metadata);$('diagnostics').href='/api/diagnostics/'+state.mapping;$('diagnostics').hidden=false;
    status('映射已生成，正在渲染同帧预览…','busy');requestPreview();scheduleSave();
  }catch(error){if(gen===state.generation){state.mapping=null;status(error.message,'error');$('preview-state').textContent='分析失败';}}
  finally{if(gen===state.generation){state.analyzing=false;setButtons();}}
}
$('analyze').addEventListener('click',analyze);
function requestPreview() {
  const version=++state.previewVersion;
  if(!state.mapping)return;
  state.previewNext={mapping:state.mapping,strength:effectiveStrength(),skin:skinSelected(),generation:state.generation,version};
  $('preview-state').textContent='预览更新中';renderPreview();
}
async function renderPreview() {
  if(state.previewRunning || !state.previewNext)return;
  state.previewRunning=true;
  while(state.previewNext){
    const request=state.previewNext;state.previewNext=null;
    try {
      const result=await api('/api/preview',{mapping_id:request.mapping,strength:request.strength,skin_protection:request.skin});
      if(!currentPreview(request))continue;
      image('before',result.before);image('after',result.after);$('before-placeholder').hidden=true;$('after-placeholder').hidden=true;
      actualTime(result.metadata);$('preview-state').textContent=$('clone-enabled').checked?`强度 ${Math.round(result.strength*100)}`:'色彩克隆已关闭';
      let message=!$('clone-enabled').checked?'色彩克隆已关闭 · 预览和导出使用原色，保留已设置的强度。':result.strength===0?'原画预览 · 强度为 0，导出将使用原画颜色。':'预览已更新 · 导出使用相同映射与当前强度。';
      if(request.skin){
        if(request.strength===0)message+=' 人物肤色保护未运行。';
        else if(result.skin?.enabled===true && result.skin.skin_detected===true)message+=' 人物肤色保护已应用，请检查皮肤边缘与偏色。';
        else if(result.skin?.enabled===true && result.skin.skin_detected===false)message+=' 未检测到可靠皮肤，使用普通调色结果。';
        else message+=' 服务未确认肤色保护，请重新预览或关闭保护后继续。';
      }
      status(message);
    } catch(error) {if(currentPreview(request)){status(error.message,'error');$('preview-state').textContent='预览失败';clearImage('after');$('after-placeholder').hidden=false;}}
  }
  state.previewRunning=false;
}
function currentPreview(request){return request.generation===state.generation && request.mapping===state.mapping && request.strength===effectiveStrength() && request.skin===skinSelected() && request.version===state.previewVersion;}
function strengthChanged(){ syncStrength();invalidateExport();setButtons();requestPreview();scheduleSave(); }
$('strength').addEventListener('input',strengthChanged);
$('strength-number').addEventListener('input',()=>{const value=$('strength-number').value;if(value.trim()==='' || !Number.isFinite(Number(value)))return;$('strength').value=String(Math.max(0,Math.min(100,Math.round(Number(value)))));strengthChanged();});
$('strength-number').addEventListener('change',syncStrength);
$('clone-enabled').addEventListener('change',()=>{invalidateExport();setButtons();requestPreview();scheduleSave();});
$('skin-enabled').addEventListener('change',()=>{invalidateExport();setButtons();requestPreview();scheduleSave();});
$('reset').addEventListener('click',()=>{$('strength').value=0;strengthChanged();});
function scheduleSave(){
  clearTimeout(state.saveTimer); if(!state.mapping)return;
  state.saveNext={generation:state.generation,version:++state.saveVersion,body:{mapping_id:state.mapping,strength:Number($('strength').value)/100,output_settings:{kind:$('format').value,clone_enabled:$('clone-enabled').checked,skin_protection:skinSelected()}}};
  $('save-state').textContent='等待保存最新项目…';
  state.saveTimer=setTimeout(()=>{state.saveTimer=null;drainSaves();},350);
}
async function drainSaves(){
  if(state.saveRunning)return;
  state.saveRunning=true;
  try {
    while(state.saveNext && state.saveTimer===null){
      const request=state.saveNext;state.saveNext=null;
      if(request.generation!==state.generation || request.body.mapping_id!==state.mapping)continue;
      $('save-state').textContent='正在保存最新项目…';
      try {
        await api('/api/projects',request.body);
        if(request.version===state.saveVersion && request.generation===state.generation && request.body.mapping_id===state.mapping)$('save-state').textContent='项目已保存在本机 · 重启可恢复';
      }catch(error){if(request.version===state.saveVersion && request.generation===state.generation)$('save-state').textContent='项目未保存：'+error.message;}
    }
  }finally{state.saveRunning=false;}
}
$('format').addEventListener('change',()=>{invalidateExport();setButtons();scheduleSave();});
$('export').addEventListener('click',async()=>{
  if(cubeProtectionBlocked()){status(cubeProtectionMessage,'error');return;}
  const gen=state.generation, mapping=state.mapping, strength=effectiveStrength(), kind=$('format').value, skin=skinSelected();
  state.job='starting';state.exportStale=false;setButtons();$('job-area').hidden=false;$('result-links').hidden=true;$('progress').style.width='0%';$('job-status').textContent='正在创建导出任务…';
  try {
    const job=await api('/api/export',{mapping_id:mapping,media_id:state.target.id,strength,kind,skin_protection:skin});
    state.job=job.id;$('cancel').hidden=false;pollJob(job.id,gen,strength,mapping,kind,skin);
  }catch(error){state.job=null;$('job-status').textContent='导出失败：'+error.message;status(error.message,'error');setButtons();}
});
async function pollJob(id,generation,strength,mapping,kind,skin=false){
  try{
    const job=await api('/api/jobs/'+id);
    if(state.job!==id)return;
    $('progress').style.width=Math.round(job.progress*100)+'%';
    const percent=Math.round(job.progress*100);
    const stale=state.exportStale || generation!==state.generation || mapping!==state.mapping || strength!==effectiveStrength() || kind!==$('format').value || skin!==skinSelected();
    $('job-status').textContent=`${job.status==='queued'?'等待导出':job.status==='running'?'正在导出':'导出'} · ${job.kind.toUpperCase()} · 强度 ${Math.round(strength*100)}%${job.status==='running'?' · '+percent+'%':''}${stale?'（先前设置，可取消）':''}`;
    if(job.status==='done'){
      $('job-status').textContent=`导出完成 · ${job.kind.toUpperCase()} · 强度 ${Math.round(strength*100)}%`;
      if(!stale && job.kind===kind){$('result-link').href=job.result_url;$('report-link').href=job.report_url;$('result-links').hidden=false;}else{$('job-status').textContent='先前设置的导出已结束（设置已更改，请按当前设置重新导出）';$('result-links').hidden=true;}
    }else if(job.status==='error'){$('job-status').textContent='导出失败：'+job.error;status('导出未完成：'+job.error,'error');}
    else if(job.status==='cancelled')$('job-status').textContent='导出已取消 · 未发布完成文件';
    else {setTimeout(()=>pollJob(id,generation,strength,mapping,kind,skin),500);return;}
    state.job=null;$('cancel').hidden=true;setButtons();
  }catch(error){$('job-status').textContent=error.message+' 将继续检查任务状态。';setTimeout(()=>pollJob(id,generation,strength,mapping,kind,skin),1800);}
}
$('cancel').addEventListener('click',async()=>{if(!state.job || state.job==='starting')return;$('cancel').disabled=true;try{await api('/api/jobs/'+state.job+'/cancel',{});}catch(error){status(error.message,'error');}finally{$('cancel').disabled=false;}});
async function loadDemo(kind){
  invalidate('正在载入受控示例…');const gen=state.generation;++state.uploading;setButtons();
  try{const demo=await api('/api/demo',{kind});if(gen!==state.generation)return;++state.uploadVersion.target;++state.uploadVersion.reference;showMedia('target',demo.target);showMedia('reference',demo.reference);await Promise.all([showFrame('target'),showFrame('reference')]);if(gen===state.generation)status('示例已就绪，即将自动分析。');}
  catch(error){if(gen===state.generation)status(error.message,'error');}finally{--state.uploading;setButtons();autoAnalyze();}
}
$('demo').addEventListener('click',()=>loadDemo('image'));
$('video-demo').addEventListener('click',()=>loadDemo('video'));
async function restore(){
  const gen=state.generation, previewVersion=state.previewVersion;
  try{
    const project=await api('/api/projects/latest');if(gen!==state.generation || previewVersion!==state.previewVersion || state.target || state.reference)return;
    showMedia('target',project.target);showMedia('reference',project.reference);$('frame-time').value=project.analysis_time;
    state.mapping=project.mapping_id;$('strength').value=Math.round(project.strength*100);$('format').value=project.output_settings.kind;
    $('clone-enabled').checked=project.output_settings.clone_enabled!==false;syncStrength();
    $('skin-enabled').checked=project.output_settings.skin_protection===true;
    $('diagnostics').href='/api/diagnostics/'+state.mapping;$('diagnostics').hidden=false;setButtons();requestPreview();
    await Promise.all([showFrame('target'),showFrame('reference')]);if(gen===state.generation)$('save-state').textContent='已恢复上次项目 · 映射缓存有效';
  }catch(error){if(gen===state.generation && !state.target){$('save-state').textContent='尚无可恢复的有效项目';if(!error.message.includes('不存在'))status('上次项目无法恢复，请重新导入与分析。','error');}}
}
setButtons();restore();
