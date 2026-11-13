const fs=require('node:fs'),path=require('node:path'),{performance}=require('node:perf_hooks');
const {manualPixels}=require('./spatial_manual_preview.js');
const dir=process.argv[2],cases=JSON.parse(fs.readFileSync(path.join(dir,'cases.json')));
const before=new Uint8ClampedArray(fs.readFileSync(path.join(dir,'before.rgba'))),full=new Uint8ClampedArray(fs.readFileSync(path.join(dir,'full.rgba')));
const results=[];let max=0,sum=0,count=0;
for(const [index,item] of cases.entries()){
 const expected=fs.readFileSync(path.join(dir,`expected-${index}.rgba`)),start=performance.now(),actual=manualPixels(before,full,item.strength,item.settings);
 let maximum=0,total=0,changed=0;
 for(let i=0;i<actual.length;i++){if(i%4===3){if(actual[i]!==255)throw Error('Nonopaque output');continue;}const d=Math.abs(actual[i]-expected[i]);maximum=Math.max(maximum,d);total+=d;if(d)changed++;}
 const channels=actual.length/4*3,mean=total/channels;max=Math.max(max,maximum);sum+=total;count+=channels;
 results.push({name:item.name,max_code:maximum,mean_code:mean,changed_channels:changed,seconds:(performance.now()-start)/1000});
 if(maximum>1||mean>.01)throw Error('Manual arithmetic gate failed: '+JSON.stringify(results.at(-1)));
}
const speed=[];
for(const name of ['basic','hsl','curves','combined']){
 const item=cases.find(x=>x.name===name),a=new Uint8ClampedArray(960*640*4),b=new Uint8ClampedArray(a.length);
 for(let i=0;i<a.length;i++){a[i]=before[i%before.length];b[i]=full[i%full.length];}
 const times=[];for(let i=0;i<5;i++){const start=performance.now();manualPixels(a,b,item.strength,item.settings);times.push((performance.now()-start)/1000);}
 speed.push({name,dimensions:[960,640],seconds:times});
}
console.log(JSON.stringify({passed:true,channels:count,max_code:max,mean_code:sum/count,cases:results,kernel_benchmarks:speed,scope:'Node numeric kernel; excludes decode, Canvas encoding and browser paint'}));
