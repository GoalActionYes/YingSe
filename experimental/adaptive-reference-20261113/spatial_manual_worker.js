/* Dedicated preview worker: never reads files, performs requests, or changes exports. */
'use strict';
importScripts('/spatial-manual-preview.js');
let basis=null;
self.onmessage=event=>{
 const message=event.data;
 try{
  if(message.type==='basis'){
   const {width,height,before,full}=message;
   if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>480||height>320||!(before instanceof Uint8ClampedArray)||!(full instanceof Uint8ClampedArray)||before.length!==width*height*4||full.length!==before.length)throw Error('Invalid worker basis');
   basis={width,height,before,full};return;
  }
  if(message.type!=='render'||!basis||!Number.isSafeInteger(message.id))throw Error('Invalid worker request');
  const pixels=self.SpatialManualPixels(basis.before,basis.full,message.strength,message.adjustments);
  self.postMessage({type:'result',id:message.id,width:basis.width,height:basis.height,pixels},[pixels.buffer]);
 }catch(error){basis=null;self.postMessage({type:'error',id:message?.id});}
};
