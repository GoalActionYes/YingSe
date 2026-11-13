/* Real Node thread executes the shipped browser-worker script via a small host bridge.
   This checks thread/protocol behavior, not browser Canvas or visual performance. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {Worker}=require('node:worker_threads'),{performance}=require('node:perf_hooks');
const {manualPixels}=require('./spatial_manual_preview.js'),{workerBasis}=require('./spatial_strength_preview.js');
async function main(){
 const out=process.argv[2],httpDir=process.argv[3],cases=JSON.parse(fs.readFileSync(path.join(out,'cases.json')));
 const kernel=fs.readFileSync(path.join(__dirname,'spatial_manual_preview.js'),'utf8'),program=fs.readFileSync(path.join(__dirname,'spatial_manual_worker.js'),'utf8');
 const host=`const {parentPort,workerData}=require('node:worker_threads');globalThis.self=globalThis;globalThis.importScripts=p=>{if(p!=='/spatial-manual-preview.js')throw Error('Unexpected import');new Function('module','window',workerData.kernel)(undefined,undefined);};self.postMessage=(m,t)=>parentPort.postMessage(m,t);new Function(workerData.program)();parentPort.on('message',data=>self.onmessage({data}));`;
 const worker=new Worker(host,{eval:true,workerData:{kernel,program}});
 let heartbeats=0;const timer=setInterval(()=>heartbeats++,2),pending=new Map();let id=0;
 worker.on('message',message=>{const p=pending.get(message.id);if(!p)return;p.resolve(message);pending.delete(message.id);});
 worker.on('error',error=>{for(const p of pending.values())p.reject(error);pending.clear();});
 function request(message){return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>{pending.delete(message.id);reject(Error('Worker timeout'));},10000);pending.set(message.id,{resolve:m=>{clearTimeout(timeout);resolve(m);},reject:e=>{clearTimeout(timeout);reject(e);}});worker.postMessage(message);});}
 const a=new Uint8ClampedArray(960*640*4),b=new Uint8ClampedArray(a.length);
 for(let i=0;i<a.length;i+=4){a.set([i%256,(i/4)%256,90,255],i);b.set([210,50,(i/8)%256,255],i);}
 const reduced=workerBasis({before:a,full:b,width:960,height:640});assert.equal(reduced.width,480);assert.equal(reduced.height,320);
 const checks=[];
 try{
  worker.postMessage(reduced);
  for(const name of ['basic','hsl','curves','combined']){
   const item=cases.find(c=>c.name===name),expected=manualPixels(reduced.before,reduced.full,item.strength,item.settings),times=[];
   for(let n=0;n<8;n++){
    const start=performance.now(),response=await request({type:'render',id:++id,strength:item.strength,adjustments:item.settings});times.push((performance.now()-start)/1000);
    assert.equal(response.type,'result');assert.equal(response.width,480);assert.equal(response.height,320);assert.deepEqual(response.pixels,expected);
   }checks.push({name,roundtrip_seconds:times});
  }
  const httpChecks=[];
  if(httpDir){
   for(const item of JSON.parse(fs.readFileSync(path.join(httpDir,'cases.json')))){
    const load=kind=>new Uint8ClampedArray(fs.readFileSync(path.join(httpDir,item.prefix+'-'+kind+'.rgba')));
    const input=workerBasis({before:load('before'),full:load('full'),width:item.width,height:item.height}),final=load('after');
    const expected=workerBasis({before:final,full:final,width:item.width,height:item.height}).before;
    worker.postMessage(input);const actual=await request({type:'render',id:++id,strength:item.strength,adjustments:item.settings});assert.equal(actual.type,'result');
    let max=0,total=0;for(let i=0;i<expected.length;i++){const error=Math.abs(actual.pixels[i]-expected[i]);max=Math.max(max,error);total+=error;}
    const mean=total/(expected.length/4*3);assert.ok(max<=1&&mean<=.01,'HTTP final preview differs: '+item.prefix);
    httpChecks.push({name:item.prefix,max_code:max,mean_code:mean,dimensions:[actual.width,actual.height]});
   }
  }
  const invalid=await request({type:'render',id:++id,strength:.5,adjustments:{version:99}});assert.equal(invalid.type,'error');
  const after=await request({type:'render',id:++id,strength:.5,adjustments:{}});assert.equal(after.type,'error');
  worker.postMessage(reduced);assert.equal((await request({type:'render',id:++id,strength:.5,adjustments:{}})).type,'result');
  assert.ok(heartbeats>10,'Main event loop did not keep advancing');
  const report={passed:true,synthetic_frames:32,dimensions:[480,320],all_synthetic_results_match_direct_kernel:true,main_thread_heartbeats:heartbeats,invalid_parameters_clear_basis:true,checks,http_final_checks:httpChecks,scope:'Actual Node worker thread using shipped scripts; excludes browser transfer overhead, Canvas, encoding and display'};
  fs.writeFileSync(path.join(out,'worker-results.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }finally{clearInterval(timer);await worker.terminate();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
