const {test}=require('node:test');
const assert=require('node:assert/strict');
const {manualPixels}=require('./spatial_manual_preview.js');
const px=values=>new Uint8ClampedArray(values.flatMap(v=>[v,v,v,255]));
test('manual preview preserves strength endpoints, disabled groups, and opaque alpha',()=>{
 const a=px([0,127,255]),b=px([255,128,0]);
 assert.deepEqual([...manualPixels(a,b,0,{})],[...a]);
 assert.deepEqual([...manualPixels(a,b,1,{})],[...b]);
 assert.deepEqual([...manualPixels(a,b,.5,{})],[128,128,128,255,128,128,128,255,128,128,128,255]);
 assert.deepEqual([...manualPixels(a,b,1,{basic:{enabled:false,brightness:100}})],[...b]);
});
test('curves run after an unrounded strength mixture and compose master then channel',()=>{
 const a=px([0]),b=px([1]);
 const out=manualPixels(a,b,.5,{curves:{rgb:[[0,0],[1,1]],r:[[0,0],[1,.5]]}});
 assert.deepEqual([...out],[0,0,0,255]);
 assert.deepEqual([...manualPixels(px([255]),px([255]),1,{curves:{rgb:[[0,0],[1,.5]],r:[[0,0],[1,.5]]}})],[64,128,128,255]);
});
test('malformed and unsupported state fails safely rather than silently dropping adjustments',()=>{
 const a=px([20]);
 for(const state of [{foo:1},{version:2},{version:null},{basic:null},{basic:{brightness:null}},{basic:{enabled:null}},{basic:{brightness:101}},{basic:{enabled:1}},{hsl:{red:[0,NaN,0]}},{curves:{rgb:[[0,0],[0,1]]}}])assert.throws(()=>manualPixels(a,a,.5,state));
 for(const s of [-1,true,NaN])assert.throws(()=>manualPixels(a,a,s,{}));
 assert.throws(()=>manualPixels(new Uint8ClampedArray([0,0,0,0]),a,.5,{}));
});
