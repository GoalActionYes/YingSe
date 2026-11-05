// Execute the actual app.js with controlled network completion and DOM events.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.resolve(__dirname, '../../reference_color_app/web/app.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function workspace(options={}) {
  const optionsConfig=options;
  const elements = new Map(), timers = new Map(); let nextTimer=0;
  function element(id) {
    if (!elements.has(id)) elements.set(id, {id, value:id==='strength'||id==='strength-number'?'80':id==='format'?'png':'0', checked:id==='clone-enabled', dataset:{}, hidden:true, disabled:false, textContent:'', className:'', style:{width:'0%'}, firstChild:{textContent:''}, listeners:{},
      addEventListener(type, callback){this.listeners[type]=callback;},
      setAttribute(key,value){this[key]=value;},scrollIntoView(){this.scrolled=true;},focus(){this.focused=true;},
      querySelector(selector){return element(id+selector);}, replaceChildren(...children){this.textContent=children.map(x=>x.textContent).join('');}, removeAttribute(key){delete this[key];}});
    return elements.get(id);
  }
  const saves=[], published=[], jobs=[], calls=[];
  const response=(data,ok=true)=>({ok,json:async()=>data});
  const context=vm.createContext({document:{getElementById:element, createTextNode:text=>({textContent:text}),createElement:()=>({textContent:''})},console,
    setTimeout(callback){const id=++nextTimer;timers.set(id,callback);return id;},clearTimeout(id){timers.delete(id);},
    fetch:async(url,options)=>{
      calls.push({url,options});
      if(url==='/api/projects/latest')return optionsForProject();
      if(optionsConfig.errors?.[url])return response({error:optionsConfig.errors[url]},false);
      const handled=optionsHandler(url,options);if(handled!==undefined)return response(await handled);
      if(url==='/api/projects')return new Promise(resolve=>{const body=JSON.parse(options.body);saves.push({body,finish(){published.push(body);resolve(response({project_id:'p'+saves.length}));}});});
      if(url.startsWith('/api/jobs/'))return new Promise(resolve=>jobs.push({finish:job=>resolve(response(job))}));
      return response({});
    }});
  vm.runInContext(source,context);
  function optionsForProject(){return options.project?response(options.project):response({error:'项目不存在'},false);}
  function optionsHandler(url,request){return options.handle?.(url,request);}
  return {element, saves, published, jobs, calls, run:code=>vm.runInContext(code,context), timers(){const callbacks=[...timers.values()];timers.clear();for(const callback of callbacks)callback();}};
}

test('ten-bit SDR import and output disclose the RGB8 compatibility conversion',async()=>{
  const ui=workspace();await flush();
  ui.run("showMedia('target',{id:'ten-bit',name:'camera.MP4',kind:'video',metadata:{width:3840,height:2160,duration_seconds:7.2,precision_conversion:true,component_bit_depths:[10,10,10]}})");
  assert.match(ui.element('target-meta').textContent,/10位 SDR → 8位处理/);
  assert.match(ui.element('export-note').textContent,/MP4输出为8位/);
  assert.equal(ui.element('format').value,'mp4');
  ui.run("showMedia('target',{id:'eight-bit',name:'normal.MP4',kind:'video',metadata:{width:1920,height:1080,duration_seconds:1,precision_conversion:false,component_bit_depths:[8,8,8]}})");
  assert.doesNotMatch(ui.element('target-meta').textContent,/10位/);
  assert.doesNotMatch(ui.element('export-note').textContent,/输入为10位/);
});

test('failed replacement clears the previous media dimensions and precision description',async()=>{
  const ui=workspace({errors:{'/api/media':'暂不支持 HDR 视频'}});await flush();
  ui.run("showMedia('target',{id:'old',name:'photo.png',kind:'image',metadata:{width:2048,height:2048}})");
  await ui.run("upload('target',{name:'new-HDR.MP4',size:100})");
  assert.equal(ui.run('state.target'),null);
  assert.doesNotMatch(ui.element('target-meta').textContent,/2048/);
  assert.match(ui.element('target-meta').textContent,/重试/);
  assert.match(ui.element('status').textContent,/HDR/);
});

test('autosave serializes pending writes and eventually persists only the newest snapshot',async()=>{
  const ui=workspace();await flush();
  ui.run("state.mapping='mapping-a';state.target={id:'target',kind:'image'};");
  ui.element('strength').value='20';ui.run('scheduleSave()');ui.timers();await flush();
  assert.equal(ui.saves.length,1);
  ui.element('strength').value='80';ui.run('scheduleSave()');ui.timers();await flush();
  assert.equal(ui.saves.length,1,'newer save must wait for older POST to finish');
  ui.saves[0].finish();await flush();
  assert.equal(ui.saves.length,2);assert.equal(ui.saves[1].body.strength,.8);
  assert.doesNotMatch(ui.element('save-state').textContent,/项目已保存在/,'old completion cannot claim newest state was saved');
  ui.saves[1].finish();await flush();
  assert.equal(ui.published.at(-1).strength,.8);assert.match(ui.element('save-state').textContent,/项目已保存在/);
});

test('a changed generation queues the current mapping after an older save completes',async()=>{
  const ui=workspace();await flush();ui.run("state.mapping='old-mapping';");
  ui.element('strength').value='20';ui.run('scheduleSave()');ui.timers();await flush();
  ui.run("invalidate();state.mapping='new-mapping';");ui.element('strength').value='90';ui.run('scheduleSave()');ui.timers();await flush();
  assert.equal(ui.saves.length,1);
  ui.saves[0].finish();await flush();
  assert.equal(ui.saves.length,2);assert.equal(ui.saves[1].body.mapping_id,'new-mapping');
  assert.doesNotMatch(ui.element('save-state').textContent,/项目已保存在/);
  ui.saves[1].finish();await flush();assert.equal(ui.published.at(-1).mapping_id,'new-mapping');
});

test('format or input invalidation clears the entire completed export presentation',async()=>{
  const ui=workspace();await flush();ui.run("state.mapping='m';state.target={id:'t',kind:'image'};");
  const completed=()=>{ui.element('job-area').hidden=false;ui.element('result-links').hidden=false;ui.element('job-status').textContent='导出完成 · PNG';ui.element('progress').style.width='100%';};
  completed();ui.element('format').value='cube';ui.element('format').listeners.change();
  assert.equal(ui.element('result-links').hidden,true);assert.equal(ui.element('job-area').hidden,true);
  assert.equal(ui.element('job-status').textContent,'');assert.equal(ui.element('progress').style.width,'0%');
  completed();ui.run('invalidate()');
  assert.equal(ui.element('job-area').hidden,true);assert.equal(ui.element('job-status').textContent,'');assert.equal(ui.element('progress').style.width,'0%');
});

test('an old-format job retains cancellation but cannot publish as the current result',async()=>{
  const ui=workspace();await flush();ui.run("state.mapping='m';state.target={id:'t',kind:'image'};state.job='j';");
  ui.element('job-area').hidden=false;ui.element('cancel').hidden=false;
  ui.run("pollJob('j',state.generation,.8,'m','png')");await flush();
  ui.element('format').value='cube';ui.element('format').listeners.change();
  assert.equal(ui.element('job-area').hidden,false);assert.equal(ui.element('cancel').hidden,false);
  ui.jobs[0].finish({id:'j',kind:'png',status:'done',progress:1,result_url:'/old.png',report_url:'/old.json'});await flush();
  assert.equal(ui.element('result-links').hidden,true,'old PNG must not appear under selected cube format');
  assert.match(ui.element('job-status').textContent,/设置已更改/);
});

const media=(id,kind='image')=>({id,kind,name:id+(kind==='video'?'.mp4':'.png'),metadata:{width:960,height:540,duration_seconds:8}});
const preview=request=>({before:'original-png',after:request.strength===0?'original-png':'colored-png',strength:request.strength,metadata:{}});
function ordinaryApi(url,options){
  const body=options?.headers?.['Content-Type']==='application/json'?JSON.parse(options.body):null;
  if(url==='/api/frame')return {image:'frame-'+body.media_id,metadata:{actual_frame_pts_seconds:body.time_seconds}};
  if(url==='/api/analyze')return {mapping_id:'new-mapping',metadata:{}};
  if(url==='/api/preview')return preview(body);
}

test('selecting the second real input automatically analyzes and previews once',async()=>{
  const ui=workspace({handle:(url,options)=>url==='/api/media'?media(decodeURIComponent(options.headers['X-Filename'])):ordinaryApi(url,options)});await flush();
  await ui.run("upload('target',{name:'target',size:12})");await flush();
  assert.equal(ui.calls.filter(x=>x.url==='/api/analyze').length,0);
  await ui.run("upload('reference',{name:'reference',size:12})");await flush();
  assert.equal(ui.calls.filter(x=>x.url==='/api/analyze').length,1);
  assert.equal(ui.element('after').src,'colored-png');assert.equal(ui.element('export').disabled,false);
  assert.match(ui.element('export-note').textContent,/导入视频后可导出 MP4/);
});

test('numeric intensity clamps and synchronizes; disabling preserves it for preview, export and saved state',async()=>{
  const ui=workspace({handle:(url,options)=>url==='/api/export'?{id:'off-export'}:ordinaryApi(url,options)});await flush();
  ui.run("state.mapping='m';state.target={id:'target',kind:'image'};setButtons();");
  ui.element('strength-number').value='125';ui.element('strength-number').listeners.input();await flush();
  assert.equal(ui.element('strength').value,'100');assert.equal(ui.element('strength-number').value,'100');
  ui.element('strength-number').value='57';ui.element('strength-number').listeners.input();await flush();
  ui.element('clone-enabled').checked=false;ui.element('clone-enabled').listeners.change();await flush();
  assert.equal(ui.element('strength').value,'57');assert.equal(ui.element('strength-number').disabled,true);
  assert.equal(ui.element('after').src,'original-png');
  await ui.element('export').listeners.click();
  const exportBody=JSON.parse(ui.calls.find(x=>x.url==='/api/export').options.body);assert.equal(exportBody.strength,0);
  ui.timers();await flush();assert.equal(ui.saves.at(-1).body.strength,.57);assert.equal(ui.saves.at(-1).body.output_settings.clone_enabled,false);
  ui.element('clone-enabled').checked=true;ui.element('clone-enabled').listeners.change();await flush();
  assert.equal(ui.element('strength-number').disabled,false);assert.equal(ui.element('after').src,'colored-png');
  assert.equal(JSON.parse(ui.calls.filter(x=>x.url==='/api/preview').at(-1).options.body).strength,.57);
});

test('restoring a disabled clone keeps the slider but renders original; old projects enable clone',async()=>{
  const project={target:media('target'),reference:media('reference'),mapping_id:'m',strength:.63,analysis_time:0,output_settings:{kind:'png',clone_enabled:false}};
  const ui=workspace({project,handle:ordinaryApi});await flush();
  assert.equal(ui.element('clone-enabled').checked,false);assert.equal(ui.element('strength-number').value,63);
  assert.equal(ui.element('strength').disabled,true);assert.equal(ui.element('after').src,'original-png');
  const old=workspace({project:{...project,output_settings:{kind:'png'}},handle:ordinaryApi});await flush();
  assert.equal(old.element('clone-enabled').checked,true);assert.equal(old.element('after').src,'colored-png');
});

test('video inputs choose MP4 and applying the analysis frame regenerates mapping; display switches are presentation only',async()=>{
  const ui=workspace({handle:ordinaryApi});await flush();
  ui.run(`showMedia('target',${JSON.stringify(media('clip','video'))});showMedia('reference',${JSON.stringify(media('ref'))});`);
  assert.equal(ui.element('format').value,'mp4');assert.equal(ui.element('format').querySelector('[value="mp4"]').disabled,false);
  assert.equal(ui.element('format').querySelector('[value="png"]').disabled,true);
  ui.element('frame-time').value='2.4';ui.element('frame-time').listeners.input();
  await ui.element('frame-load').listeners.click();await flush();
  assert.equal(JSON.parse(ui.calls.find(x=>x.url==='/api/analyze').options.body).time_seconds,2.4);
  ui.element('view-original').listeners.click();assert.equal(ui.element('comparison').dataset.view,'original');
  ui.element('view-compare').listeners.click();assert.equal(ui.element('comparison').dataset.view,'compare');
  assert.equal(ui.calls.filter(x=>x.url==='/api/analyze').length,1);
  ui.element('top-export').listeners.click();assert.equal(ui.element('export-area').focused,true);
});

test('a superseded upload completing last still auto analyzes the newest valid media pair',async()=>{
  const pending=[];
  const ui=workspace({handle:(url,options)=>url==='/api/media'?new Promise(resolve=>pending.push({name:decodeURIComponent(options.headers['X-Filename']),resolve})):ordinaryApi(url,options)});await flush();
  ui.run(`showMedia('reference',${JSON.stringify(media('reference'))});`);
  const old=ui.run("upload('target',{name:'old',size:12})");const current=ui.run("upload('target',{name:'new',size:12})");
  pending[1].resolve(media('new'));await current;assert.equal(ui.calls.filter(x=>x.url==='/api/analyze').length,0);
  pending[0].resolve(media('old'));await old;await flush();
  assert.equal(ui.calls.filter(x=>x.url==='/api/analyze').length,1);
  assert.equal(JSON.parse(ui.calls.find(x=>x.url==='/api/analyze').options.body).target_id,'new');
});

test('an in-flight colored preview cannot overwrite a newly disabled clone',async()=>{
  const previews=[];
  const ui=workspace({handle:(url,options)=>url==='/api/preview'?new Promise(resolve=>previews.push({body:JSON.parse(options.body),resolve})):ordinaryApi(url,options)});await flush();
  ui.run("state.mapping='m';requestPreview();");await flush();assert.equal(previews[0].body.strength,.8);
  ui.element('clone-enabled').checked=false;ui.element('clone-enabled').listeners.change();
  previews[0].resolve(preview(previews[0].body));await flush();
  assert.notEqual(ui.element('after').src,'colored-png');assert.equal(previews[1].body.strength,0);
  previews[1].resolve(preview(previews[1].body));await flush();assert.equal(ui.element('after').src,'original-png');
});

test('a superseded demo completing after a new upload triggers analysis of the current pair',async()=>{
  let finishDemo;
  const ui=workspace({handle:(url,options)=>url==='/api/demo'?new Promise(resolve=>{finishDemo=resolve;}):url==='/api/media'?media('new-target'):ordinaryApi(url,options)});await flush();
  ui.run(`showMedia('target',${JSON.stringify(media('old-target'))});showMedia('reference',${JSON.stringify(media('current-reference'))});`);
  const oldDemo=ui.run("loadDemo('image')");
  await ui.run("upload('target',{name:'new-target',size:12})");
  assert.equal(ui.calls.filter(x=>x.url==='/api/analyze').length,0,'analysis must wait for pending demo operation');
  finishDemo({target:media('stale-demo-target'),reference:media('stale-demo-reference')});await oldDemo;await flush();
  const analyses=ui.calls.filter(x=>x.url==='/api/analyze');assert.equal(analyses.length,1);
  const body=JSON.parse(analyses[0].options.body);assert.equal(body.target_id,'new-target');assert.equal(body.reference_id,'current-reference');
  assert.equal(ui.element('after').src,'colored-png');assert.equal(ui.element('export').disabled,false);
});

test('skin protection is unchecked in actual HTML and explicitly false in ordinary previews',async()=>{
  const html=fs.readFileSync(path.resolve(__dirname,'../../reference_color_app/web/index.html'),'utf8');
  assert.match(html,/<title>映色 · 肤色保护实验版<\/title>/);
  const checkbox=html.match(/<input[^>]+id="skin-enabled"[^>]*>/)?.[0];
  assert.ok(checkbox);assert.doesNotMatch(checkbox,/\bchecked\b/);
  const ui=workspace({handle:ordinaryApi});await flush();
  ui.run("state.mapping='m';requestPreview()");await flush();
  assert.equal(JSON.parse(ui.calls.find(x=>x.url==='/api/preview').options.body).skin_protection,false);
});

test('skin protection is saved as a boolean and restores only an explicit true value',async()=>{
  const project={target:media('target'),reference:media('reference'),mapping_id:'m',strength:.8,analysis_time:0,output_settings:{kind:'png',skin_protection:true}};
  const ui=workspace({project,handle:ordinaryApi});await flush();
  assert.equal(ui.element('skin-enabled').checked,true);
  ui.run('scheduleSave()');ui.timers();await flush();
  assert.equal(ui.saves[0].body.output_settings.skin_protection,true);ui.saves[0].finish();await flush();
  ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();ui.timers();await flush();
  assert.equal(ui.saves[1].body.output_settings.skin_protection,false);ui.saves[1].finish();
  const old=workspace({project:{...project,output_settings:{kind:'png'}},handle:ordinaryApi});await flush();
  assert.equal(old.element('skin-enabled').checked,false);
});

test('skin changes preserve serial autosaving and eventually persist the newest checkbox snapshot',async()=>{
  const ui=workspace({handle:ordinaryApi});await flush();ui.run("state.mapping='m'");
  ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();ui.timers();await flush();
  ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();ui.timers();await flush();
  assert.equal(ui.saves.length,1);assert.equal(ui.saves[0].body.output_settings.skin_protection,true);
  ui.saves[0].finish();await flush();assert.equal(ui.saves.length,2);
  assert.equal(ui.saves[1].body.output_settings.skin_protection,false);ui.saves[1].finish();await flush();
  assert.equal(ui.published.at(-1).output_settings.skin_protection,false);
});

test('unchecked-checked-unchecked preview ABA cannot display an obsolete response or obsolete error',async()=>{
  const pending=[];
  const ui=workspace({handle:(url,options)=>url==='/api/preview'?new Promise((resolve,reject)=>pending.push({body:JSON.parse(options.body),resolve,reject})):ordinaryApi(url,options)});await flush();
  ui.run("state.mapping='m';requestPreview()");await flush();
  ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();
  ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();
  pending[0].resolve({...preview(pending[0].body),after:'obsolete-preview'});await flush();
  assert.notEqual(ui.element('after').src,'obsolete-preview');assert.equal(pending.length,2);
  pending[1].resolve({...preview(pending[1].body),after:'current-preview'});await flush();
  assert.equal(ui.element('after').src,'current-preview');
  ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();await flush();
  ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();
  pending[2].reject(Error('过时的模型错误'));await flush();
  assert.doesNotMatch(ui.element('status').textContent,/过时/);assert.equal(ui.element('after').src,'current-preview');
  pending[3].resolve(preview(pending[3].body));await flush();
});

test('skin response diagnostics distinguish detected, empty and bypassed protection',async()=>{
  let skin={enabled:true,skin_detected:false};
  const ui=workspace({handle:(url,options)=>url==='/api/preview'?{...preview(JSON.parse(options.body)),skin}:ordinaryApi(url,options)});await flush();
  ui.run("state.mapping='m'");ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();await flush();
  assert.match(ui.element('status').textContent,/未检测到可靠皮肤/);assert.doesNotMatch(ui.element('status').textContent,/保护已应用/);
  skin={enabled:true,skin_detected:true};ui.run('requestPreview()');await flush();
  assert.match(ui.element('status').textContent,/肤色保护已应用/);
  ui.element('clone-enabled').checked=false;ui.element('clone-enabled').listeners.change();await flush();
  assert.match(ui.element('status').textContent,/保护未运行/);assert.doesNotMatch(ui.element('status').textContent,/保护已应用/);
  const body=JSON.parse(ui.calls.filter(x=>x.url==='/api/preview').at(-1).options.body);
  assert.equal(body.skin_protection,true);assert.equal(body.strength,0);assert.equal(ui.element('skin-enabled').checked,true);
});

test('skin inference errors remain explicit and turning protection off recovers baseline preview',async()=>{
  const errors={'/api/preview':'皮肤识别失败：模型缺失，请关闭保护'};
  const ui=workspace({errors,handle:ordinaryApi});await flush();
  ui.run("state.mapping='m'");ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();await flush();
  assert.equal(ui.element('status').className,'status error');assert.match(ui.element('status').textContent,/皮肤识别失败/);assert.equal(ui.element('after').hidden,true);
  delete errors['/api/preview'];ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();await flush();
  assert.equal(ui.element('after').src,'colored-png');assert.equal(ui.element('after').hidden,false);
});

test('CUBE plus selected protection is blocked even with clone off, without changing user choices',async()=>{
  const ui=workspace({handle:ordinaryApi});await flush();ui.run("state.mapping='m';state.target={id:'t',kind:'image'}");
  ui.element('format').value='cube';ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();
  assert.equal(ui.element('export').disabled,true);assert.match(ui.element('export-note').textContent,/CUBE不包含人物区域保护/);
  await ui.element('export').listeners.click();assert.equal(ui.calls.filter(x=>x.url==='/api/export').length,0);
  ui.element('clone-enabled').checked=false;ui.element('clone-enabled').listeners.change();
  assert.equal(ui.element('export').disabled,true);assert.equal(ui.element('skin-enabled').checked,true);assert.equal(ui.element('format').value,'cube');
  ui.element('format').value='png';ui.element('format').listeners.change();
  assert.equal(ui.element('export').disabled,false);assert.equal(ui.element('skin-enabled').checked,true);
});

test('protected video warns about 30fps and export carries the selected boolean',async()=>{
  const ui=workspace({handle:(url,options)=>url==='/api/export'?{id:'skin-export'}:ordinaryApi(url,options)});await flush();
  ui.run(`showMedia('target',${JSON.stringify(media('clip','video'))});state.mapping='m'`);
  ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();await flush();
  assert.match(ui.element('export-note').textContent,/30fps/);assert.match(ui.element('export-note').textContent,/较慢/);
  await ui.element('export').listeners.click();
  const body=JSON.parse(ui.calls.find(x=>x.url==='/api/export').options.body);assert.equal(body.skin_protection,true);assert.equal(body.kind,'mp4');
  ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();
  assert.doesNotMatch(ui.element('export-note').textContent,/30fps/);
});

test('skin toggles invalidate completed links and an in-flight old job cannot republish them',async()=>{
  const ui=workspace({handle:ordinaryApi});await flush();ui.run("state.mapping='m';state.target={id:'t',kind:'image'}");
  ui.element('result-links').hidden=false;ui.element('job-area').hidden=false;
  ui.element('skin-enabled').checked=true;ui.element('skin-enabled').listeners.change();await flush();
  assert.equal(ui.element('result-links').hidden,true);assert.equal(ui.element('job-area').hidden,true);
  ui.run("state.job='j';pollJob('j',state.generation,.8,'m','png',true)");await flush();
  ui.element('skin-enabled').checked=false;ui.element('skin-enabled').listeners.change();
  ui.jobs[0].finish({id:'j',kind:'png',status:'done',progress:1,result_url:'/protected.png',report_url:'/protected.json'});await flush();
  assert.equal(ui.element('result-links').hidden,true);assert.match(ui.element('job-status').textContent,/设置已更改/);
});
