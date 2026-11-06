const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../reference_color_app/web/app.js'),'utf8');
test('preview scheduling updates during continuous dragging and coalesces bursts',()=>{
 const start=source.indexOf('function schedulePreview()');assert.notEqual(start,-1);
 const code=source.slice(start,source.indexOf('function requestPreview()',start));
 let ticks=[],calls=0;const context=vm.createContext({manualTimer:null,setTimeout:fn=>{ticks.push(fn);return ticks.length;},requestPreview:()=>++calls});
 vm.runInContext(code,context);
 for(let i=0;i<20;i++)context.schedulePreview();
 assert.equal(ticks.length,1);ticks.shift()();assert.equal(calls,1);
 context.schedulePreview();assert.equal(ticks.length,1);ticks.shift()();assert.equal(calls,2);
});
for(const stale of [false,true])test('failed preview preserves last frame; stale errors ignored: '+stale,async()=>{
 const start=source.indexOf('async function renderPreview()');
 const code=source.slice(start,source.indexOf('function strengthChanged()',start));
 let cleared=0,messages=0;const nodes={};
 const context=vm.createContext({state:{previewRunning:false,previewNext:{mapping:'m',generation:1,manualVersion:0,strength:stale?.4:.8},mapping:'m',generation:1},manualVersion:0,
  api:async()=>{throw Error('network');},effectiveStrength:()=>.8,$:id=>nodes[id]??={},status:()=>++messages,clearImage:()=>++cleared});
 vm.runInContext(code,context);await context.renderPreview();
 assert.equal(cleared,0);assert.equal(messages,stale?0:1);assert.equal(context.state.previewRunning,false);
});
