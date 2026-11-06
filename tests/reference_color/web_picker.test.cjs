const {test}=require('node:test');const assert=require('node:assert/strict');const vm=require('node:vm');const fs=require('node:fs');const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../../reference_color_app/web/adjustments.js'),'utf8');
const context=vm.createContext({});vm.runInContext(source.slice(source.indexOf('function imageCoordinates'),source.indexOf('// User controls')),context);
test('picker excludes letterbox and handles scaled landscape/portrait',()=>{
 const image={naturalWidth:4000,naturalHeight:2000,getBoundingClientRect:()=>({left:10,top:20,width:800,height:600})};
 assert.equal(context.imageCoordinates({clientX:410,clientY:50},image),null);
 let p=context.imageCoordinates({clientX:410,clientY:320},image);assert.equal(p.x,.5);assert.equal(p.y,.5);
 image.naturalWidth=1000;image.naturalHeight=2000;
 assert.equal(context.imageCoordinates({clientX:50,clientY:320},image),null);
 p=context.imageCoordinates({clientX:410,clientY:320},image);assert.equal(p.x,.5);assert.equal(p.y,.5);
 image.naturalWidth=0;assert.equal(context.imageCoordinates({clientX:410,clientY:320},image),null);
});
