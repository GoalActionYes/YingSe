(function(root){
 'use strict';
 const tables=new Map(),normalized=Float32Array.from({length:256},(_,i)=>i/255);
 function blendPixels(before,full,strength){
  if(typeof strength!=='number'||!Number.isFinite(strength)||strength<0||strength>1)throw Error('Invalid strength');
  if(!(before instanceof Uint8ClampedArray)||!(full instanceof Uint8ClampedArray)||!before.length||before.length%4||before.length!==full.length)throw Error('Invalid RGBA basis');
  const s=Math.fround(strength);let table=tables.get(s);
  if(!table){
   table=new Uint8ClampedArray(65536);const q=Math.fround(1-s);
   for(let a=0;a<256;a++){
    const original=Math.fround(normalized[a]*q);
    for(let b=0;b<256;b++)table[(a<<8)|b]=Math.fround(Math.fround(original+Math.fround(normalized[b]*s))*255);
   }
   if(tables.size>=8)tables.delete(tables.keys().next().value);tables.set(s,table);
  }
  const result=new Uint8ClampedArray(before.length);
  for(let i=0;i<before.length;i+=4){
   if(before[i+3]!==255||full[i+3]!==255)throw Error('Opaque RGB basis required');
   result[i]=table[(before[i]<<8)|full[i]];result[i+1]=table[(before[i+1]<<8)|full[i+1]];result[i+2]=table[(before[i+2]<<8)|full[i+2]];result[i+3]=255;
  }
  return result;
 }
 function decodePng(uri){
  return new Promise((resolve,reject)=>{
   if(typeof uri!=='string'||!uri.startsWith('data:image/png;base64,')||uri.length>8*1024*1024)return reject(Error('Invalid preview PNG'));
   const image=new Image();image.onerror=()=>reject(Error('Cannot decode preview PNG'));
   image.onload=()=>{
    try{
     const width=image.naturalWidth,height=image.naturalHeight;
     if(!width||!height||width*height>1440*960)throw Error('Preview basis exceeds bounds');
     const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
     const ctx=canvas.getContext('2d',{willReadFrequently:true});if(!ctx)throw Error('Canvas unavailable');
     ctx.drawImage(image,0,0);resolve({width,height,pixels:ctx.getImageData(0,0,width,height).data});
    }catch(error){reject(error);}
   };image.src=uri;
  });
 }
 const key=(mapping,generation)=>JSON.stringify([mapping,generation]);
 const signature=s=>JSON.stringify([s.mapping,s.generation,s.strength,s.neutral,s.version]);
 class StrengthPreview{
  constructor(options){
   this.current=options.current;this.decode=options.decode||decodePng;this.target=options.target;this.onPaint=options.onPaint||(()=>{});
   this.requestFrame=options.requestFrame||(fn=>requestAnimationFrame(fn));this.cancelFrame=options.cancelFrame||(id=>cancelAnimationFrame(id));
   this.paint=options.paint||((pixels,width,height)=>{
    if(!this.canvas)this.canvas=document.createElement('canvas');
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    const ctx=this.canvas.getContext('2d');if(!ctx||!this.target)throw Error('Preview surface unavailable');
    ctx.putImageData(new ImageData(pixels,width,height),0,0);this.target.src=this.canvas.toDataURL('image/png');this.target.hidden=false;this.onPaint();
   });
   this.frame=null;this.loadSequence=0;this.clear();
  }
  cancel(){if(this.frame!==null)this.cancelFrame(this.frame);this.frame=null;}
  clear(){this.cancel();this.loadSequence++;this.key=null;this.basis=null;this.requested=false;this.serverSignature=null;}
  needsBasis(mapping,generation){
   const state=this.current(),wanted=key(mapping,generation);
   if(mapping!==state.mapping||generation!==state.generation)return false;
   if(this.key!==wanted){this.clear();this.key=wanted;}
   if(this.basis||this.requested)return false;
   this.requested=true;return true;
  }
  async accept(mapping,generation,before,basis){
   const wanted=key(mapping,generation),state=this.current();
   if(this.key!==wanted||key(state.mapping,state.generation)!==wanted)return false;
   const ticket=++this.loadSequence;
   try{
    if(!basis||basis.version!==1||!Number.isInteger(basis.width)||!Number.isInteger(basis.height)||basis.width<1||basis.height<1||basis.width*basis.height>1440*960)throw Error('Unsupported basis');
    const [a,b]=await Promise.all([this.decode(before),this.decode(basis.full)]);
    if(ticket!==this.loadSequence||this.key!==wanted||key(this.current().mapping,this.current().generation)!==wanted)return false;
    for(const image of [a,b])if(image.width!==basis.width||image.height!==basis.height||image.pixels.length!==basis.width*basis.height*4)throw Error('Basis dimensions changed');
    this.basis={before:a.pixels,full:b.pixels,width:basis.width,height:basis.height};this.requested=false;this.render();return true;
   }catch(error){if(ticket===this.loadSequence&&this.key===wanted)this.clear();return false;}
  }
  eligible(){const s=this.current();return s.mapping&&s.neutral&&this.basis&&this.key===key(s.mapping,s.generation)&&signature(s)!==this.serverSignature;}
  render(){
   if(!this.eligible()){this.cancel();return false;}
   if(this.frame!==null)return true;
   this.frame=this.requestFrame(()=>{
    this.frame=null;if(!this.eligible())return;
    try{const s=this.current(),b=this.basis;this.paint(blendPixels(b.before,b.full,s.strength),b.width,b.height);}
    catch(error){this.clear();}
   });return true;
  }
  serverApplied(){this.cancel();this.serverSignature=signature(this.current());}
 }
 if(typeof module!=='undefined'&&module.exports)module.exports={blendPixels,StrengthPreview};
 else root.SpatialStrengthPreview=StrengthPreview;
})(typeof window==='undefined'?globalThis:window);
