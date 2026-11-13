'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {performance}=require('node:perf_hooks');
const {blendPixels}=require('./spatial_strength_preview.js');
const root=process.argv[2],metadata=JSON.parse(fs.readFileSync(path.join(root,'golden.json'),'utf8'));
const read=name=>new Uint8ClampedArray(fs.readFileSync(path.join(root,name)));
const before=read('before.rgba'),full=read('full.rgba'),expected=read('expected.rgba');
let max=0,changed=0;
for(let strength=0;strength<=100;strength++){
 const actual=blendPixels(before,full,strength/100),offset=strength*before.length;
 for(let i=0;i<actual.length;i++){const delta=Math.abs(actual[i]-expected[offset+i]);if(delta)changed++;max=Math.max(max,delta);}
}
assert.ok(max<=1,'Browser arithmetic differs from backend by more than one code value');
const a=new Uint8ClampedArray(960*640*4),b=new Uint8ClampedArray(a.length);
for(let i=0;i<a.length;i+=4){a[i]=i%256;a[i+1]=(i/4)%256;a[i+2]=90;a[i+3]=255;b[i]=210;b[i+1]=50;b[i+2]=(i/8)%256;b[i+3]=255;}
const times=[];for(let repeat=0;repeat<8;repeat++){const start=performance.now();blendPixels(a,b,(repeat+1)/10);times.push(performance.now()-start);}
const report={max_rgb_code_error:max,changed_values:changed,checked_values:expected.length,all_65536_channel_pairs:true,strengths:metadata.strengths,
 numeric_blend_960x640_ms:times,scope:'Node numeric arithmetic and timing; excludes browser image decode, canvas PNG encoding, layout and paint'};
fs.writeFileSync(path.join(root,'math-results.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
