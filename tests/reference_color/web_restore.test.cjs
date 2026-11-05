const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../../reference_color_app/web/app.js'),'utf8');
const restore=source.slice(source.indexOf('async function restore(){'),source.indexOf("window.addEventListener('manual-change'"));
for(const userEdits of [false,true])test('restore preserves edits while loading: '+userEdits,async()=>{
  let resolve,restored=0,saved=0;
  const nodes={};const state={generation:0};
  const context=vm.createContext({state,manualVersion:0,
    api:()=>new Promise(r=>resolve=r),$:id=>nodes[id]??=( {value:0,checked:true}),
    showMedia:(key,value)=>state[key]=value,controls:{restore:()=>++restored},
    syncStrength:()=>{},setButtons:()=>{},requestPreview:()=>{},
    scheduleSave:()=>++saved,showFrame:async()=>{},status:message=>{throw Error(message);}});
  vm.runInContext(restore,context);const running=context.restore();
  if(userEdits)vm.runInContext('++manualVersion',context);
  resolve({target:{id:'target'},reference:{id:'reference'},analysis_time:0,mapping_id:'mapping',strength:.8,
    output_settings:{kind:'png',clone_enabled:true,adjustments:{basic:{brightness:10}}}});
  await running;
  assert.equal(restored,userEdits?0:1);assert.equal(saved,userEdits?1:0);
  assert.equal(state.mapping,'mapping');
});
