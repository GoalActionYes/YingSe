const {test}=require('node:test');
const assert=require('node:assert/strict');
const {blendPixels,StrengthPreview,workerBasis}=require('./spatial_strength_preview.js');

test('mixes opaque pixels with even rounding and rejects invalid input',()=>{
 const a=new Uint8ClampedArray([0,127,255,255]),b=new Uint8ClampedArray([255,128,0,255]);
 assert.deepEqual([...blendPixels(a,b,.5)],[128,128,128,255]);
 assert.deepEqual([...blendPixels(a,b,0)],[...a]);assert.deepEqual([...blendPixels(a,b,1)],[...b]);
 for(const strength of [-1,1.01,NaN,true])assert.throws(()=>blendPixels(a,b,strength));
 assert.throws(()=>blendPixels(a,b.subarray(0,3),.5));
 assert.throws(()=>blendPixels(new Uint8ClampedArray([0,0,0,0]),b,.5));
});

function rig(decode){
 const state={mapping:'one',generation:1,strength:.5,neutral:true,version:0},frames=new Map(),painted=[];let id=0;
 const p=new StrengthPreview({current:()=>({...state}),decode,paint:(pixels)=>painted.push([...pixels]),requestFrame:fn=>{frames.set(++id,fn);return id;},cancelFrame:n=>frames.delete(n)});
 return {state,p,painted,flush(){for(const [key,fn] of [...frames]){frames.delete(key);fn();}}};
}
const pair={version:1,width:1,height:1,full:'full'};
const decode=async uri=>({width:1,height:1,pixels:new Uint8ClampedArray(uri==='full'?[200,100,0,255]:[0,0,0,255])});
test('coalesces drag values, never replaces a final server frame, and falls back for manual changes',async()=>{
 const r=rig(decode);assert.equal(r.p.needsBasis('one',1),true);assert.equal(r.p.needsBasis('one',1),false);
 await r.p.accept('one',1,'before',pair);r.flush();assert.deepEqual(r.painted.at(-1),[100,50,0,255]);
 r.state.strength=.2;r.p.render();r.state.strength=.8;r.p.render();r.flush();assert.deepEqual(r.painted.at(-1),[160,80,0,255]);
 r.state.strength=.3;r.p.render();r.p.serverApplied();const count=r.painted.length;r.flush();assert.equal(r.painted.length,count);
 r.state.neutral=false;r.state.version++;assert.equal(r.p.render(),false);r.flush();assert.equal(r.painted.length,count);
});
test('late decoded basis cannot paint a replaced target; errors allow a retry',async()=>{
 let release;const wait=new Promise(resolve=>release=resolve),r=rig(async uri=>{await wait;return decode(uri);});
 r.p.needsBasis('one',1);const pending=r.p.accept('one',1,'before',pair);
 r.state.mapping='two';r.state.generation=2;r.p.clear();release();assert.equal(await pending,false);r.flush();assert.equal(r.painted.length,0);
 const broken=rig(async()=>{throw Error('decode failed');});broken.p.needsBasis('one',1);assert.equal(await broken.p.accept('one',1,'before',pair),false);
 assert.equal(broken.p.needsBasis('one',1),true);
});
test('basis decoding after an authoritative response never repaints that response',async()=>{
 let release;const wait=new Promise(resolve=>release=resolve),r=rig(async uri=>{await wait;return decode(uri);});
 r.p.needsBasis('one',1);const pending=r.p.accept('one',1,'before',pair);r.p.serverApplied();release();await pending;r.flush();assert.equal(r.painted.length,0);
 r.state.strength=.9;r.p.render();r.flush();assert.deepEqual(r.painted[0],[180,90,0,255]);
});

test('manual worker coalesces changes, rejects stale output, and defers to final server frame',async()=>{
 const r=rig(decode),sent=[];let worker;
 r.p.workerFactory=()=>worker={postMessage(m){sent.push(m);},terminate(){this.stopped=true;}};
 r.p.needsBasis('one',1);await r.p.accept('one',1,'before',pair);r.flush();
 r.state.neutral=false;r.state.adjustments={basic:{brightness:12}};r.state.version++;
 assert.equal(r.p.render(),true);r.flush();assert.equal(sent[0].type,'basis');const first=sent.at(-1);
 r.state.version++;r.state.adjustments={basic:{brightness:35}};r.p.render();r.flush();assert.equal(sent.length,2);
 const count=r.painted.length;
 worker.onmessage({data:{type:'result',id:first.id,width:1,height:1,pixels:new Uint8ClampedArray([1,2,3,255])}});
 assert.equal(r.painted.length,count);r.flush();const latest=sent.at(-1);assert.equal(latest.adjustments.basic.brightness,35);
 worker.onmessage({data:{type:'result',id:latest.id,width:1,height:1,pixels:new Uint8ClampedArray([4,5,6,255])}});
 assert.deepEqual(r.painted.at(-1),[4,5,6,255]);
 r.state.version++;r.p.render();r.flush();const final=sent.at(-1);r.p.serverApplied();
 worker.onmessage({data:{type:'result',id:final.id,width:1,height:1,pixels:new Uint8ClampedArray([9,9,9,255])}});
 assert.deepEqual(r.painted.at(-1),[4,5,6,255]);
 r.p.clear();assert.equal(worker.stopped,true);
});

test('manual worker failure keeps neutral preview working and falls back for adjustments',async()=>{
 const r=rig(decode);let worker;
 r.p.workerFactory=()=>worker={postMessage(){},terminate(){}};
 r.p.needsBasis('one',1);await r.p.accept('one',1,'before',pair);r.flush();r.state.neutral=false;r.state.adjustments={};r.state.version++;
 r.p.render();r.flush();worker.onerror({preventDefault(){}});assert.equal(r.p.render(),false);
 r.state.neutral=true;r.state.strength=.9;assert.equal(r.p.render(),true);r.flush();assert.deepEqual(r.painted.at(-1),[180,90,0,255]);
});

test('replaced materials terminate pending manual work and ignore its late messages',async()=>{
 const r=rig(decode);let old;
 r.p.workerFactory=()=>old={postMessage(){},terminate(){this.stopped=true;}};
 r.p.needsBasis('one',1);await r.p.accept('one',1,'before',pair);r.flush();r.state.neutral=false;r.state.adjustments={};r.state.version++;
 r.p.render();r.flush();const count=r.painted.length;
 r.state.mapping='two';r.state.generation++;r.p.clear();assert.equal(old.stopped,true);
 old.onmessage({data:{type:'result',id:1,width:1,height:1,pixels:new Uint8ClampedArray([9,9,9,255])}});old.onmessageerror();
 assert.equal(r.painted.length,count);assert.equal(r.p.manualFailed,false);
});

test('worker basis bounds landscape and portrait dimensions and keeps source byte samples',()=>{
 for(const [width,height] of [[960,640],[426,640],[1440,1],[1,960]]){
  const pixels=new Uint8ClampedArray(width*height*4);for(let i=0;i<pixels.length;i+=4)pixels.set([i%255,20,90,255],i);
  const b=workerBasis({width,height,before:pixels,full:pixels});assert.ok(b.width<=480&&b.height<=320);assert.equal(b.before.length,b.width*b.height*4);
  const sx=Math.floor(.5*width/b.width),sy=Math.floor(.5*height/b.height);assert.deepEqual([...b.before.subarray(0,4)],[...pixels.subarray((sy*width+sx)*4,(sy*width+sx)*4+4)]);
 }
});

if(process.env.SPATIAL_APP_JS){
 const fs=require('node:fs'),vm=require('node:vm'),source=fs.readFileSync(process.env.SPATIAL_APP_JS,'utf8');
 test('served enhanced script parses as a complete browser program',()=>new vm.Script(source));
 for(const stale of [true,false])test('served response keeps valid basis while respecting newer slider values: '+stale,async()=>{
  const start=source.indexOf('async function renderPreview()'),code=source.slice(start,source.indexOf('function strengthChanged()',start));
  let accepted=0,applied=0,painted=0;const nodes={},state={mapping:'m',generation:1,previewRunning:false,previewNext:{mapping:'m',generation:1,strength:stale?.2:.8,manualVersion:0}};
  const context=vm.createContext({state,manualVersion:0,fastStrength:{needsBasis:()=>true,accept:()=>accepted++,serverApplied:()=>applied++,clear(){}},
   api:async()=>({basis:pair,before:'before',after:'after',strength:.8,metadata:{}}),effectiveStrength:()=>.8,controls:{count:()=>0},
   $:id=>nodes[id]??={},image:()=>painted++,actualTime(){},status(){}});
  vm.runInContext(code,context);await context.renderPreview();assert.equal(accepted,1);assert.equal(applied,stale?0:1);assert.equal(painted,stale?0:2);
 });
 for(const replaced of [false,true])test('current API failure clears the local basis without touching a replacement: '+replaced,async()=>{
  const start=source.indexOf('async function renderPreview()'),code=source.slice(start,source.indexOf('function strengthChanged()',start));
  let cleared=0;const state={mapping:replaced?'new':'old',generation:replaced?2:1,previewRunning:false,previewNext:{mapping:'old',generation:1,strength:.2,manualVersion:0}};
  const context=vm.createContext({state,manualVersion:0,fastStrength:{needsBasis:()=>false,clear:()=>cleared++},api:async()=>{throw Error('invalid source');},effectiveStrength:()=>.8});
  vm.runInContext(code,context);await context.renderPreview();assert.equal(cleared,replaced?0:1);
 });
}
