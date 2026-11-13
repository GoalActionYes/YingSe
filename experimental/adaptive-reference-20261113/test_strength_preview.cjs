const {test}=require('node:test');
const assert=require('node:assert/strict');
const {blendPixels,StrengthPreview}=require('./spatial_strength_preview.js');

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
