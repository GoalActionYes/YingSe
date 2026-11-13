/* Local preview of the existing Python finishing chain; exports stay authoritative. */
(function(root){
 'use strict';
 const basic=['temperature','tint','saturation','vibrance','brightness','contrast','highlights','shadows','whites','blacks','brilliance'];
 const colors=['red','orange','yellow','green','cyan','blue','purple','magenta'];
 const centers=[0,30,60,120,180,240,275,315].map(v=>Math.fround(v/360));
 const f=Math.fround,clip=x=>Math.max(0,Math.min(1,x));
 const get=(obj,key,fallback)=>Object.hasOwn(obj,key)?obj[key]:fallback;
 function number(x,lo=-100,hi=100){if(typeof x!=='number'||!Number.isFinite(x)||x<lo||x>hi)throw Error('Invalid manual parameter');return x;}
 function object(x,allowed){if(!x||typeof x!=='object'||Array.isArray(x)||Object.keys(x).some(k=>!allowed.includes(k)))throw Error('Unsupported manual settings');return x;}
 function normalize(state){
  state=object(state??{},['version','basic','hsl','curves']);if(get(state,'version',1)!==1)throw Error('Unsupported manual version');
  const out={};
  for(const [group,keys] of [['basic',basic],['hsl',colors],['curves',['rgb','r','g','b']]]){
   const src=object(get(state,group,{}),['enabled',...keys]),dest={enabled:get(src,'enabled',true)};if(typeof dest.enabled!=='boolean')throw Error('Invalid enable flag');
   for(const key of keys){
    if(group==='basic')dest[key]=number(get(src,key,0));
    else if(group==='hsl'){
     const v=get(src,key,[0,0,0]);if(!Array.isArray(v)||v.length!==3)throw Error('Invalid HSL');dest[key]=v.map(x=>number(x));
    }else{
     const v=get(src,key,[[0,0],[1,1]]);
     if(!Array.isArray(v)||v.length<2||v.length>16)throw Error('Invalid curve');
     dest[key]=v.map(p=>{if(!Array.isArray(p)||p.length!==2)throw Error('Invalid curve point');return p.map(x=>number(x,0,1));});
     if(dest[key][0][0]!==0||dest[key].at(-1)[0]!==1||dest[key].some((p,i)=>i&&p[0]-dest[key][i-1][0]<.001))throw Error('Invalid curve order');
    }
   }out[group]=dest;
  }return out;
 }
 function curve(x,points){
  if(x<=0)return points[0][1];if(x>=1)return points.at(-1)[1];
  let i=1;while(points[i][0]<x)i++;
  const [a,b]=points[i-1],[c,d]=points[i];return f(b+(x-a)/(c-a)*(d-b));
 }
 function identity(points){return points.length===2&&points[0][0]===0&&points[0][1]===0&&points[1][0]===1&&points[1][1]===1;}
 function manualPixels(before,full,strength,state){
  number(strength,0,1);
  if(!(before instanceof Uint8ClampedArray)||!(full instanceof Uint8ClampedArray)||!before.length||before.length%4||before.length!==full.length)throw Error('Invalid preview pixels');
  const settings=normalize(state),p=settings.basic,s=f(strength),q=f(1-s),out=new Uint8ClampedArray(before.length);
  const activeBasic=p.enabled&&basic.some(k=>p[k]!==0),activeHsl=settings.hsl.enabled&&colors.some(k=>settings.hsl[k].some(v=>v!==0));
  const curves=settings.curves,activeCurves=curves.enabled&&['rgb','r','g','b'].some(k=>!identity(curves[k]));
  const v=Object.fromEntries(basic.map(k=>[k,p[k]/100]));
  const gain=[f(2**f(v.temperature*.35)),f(2**f(-v.tint*.2)),f(2**f(-v.temperature*.35))],tint=2**(v.tint*.1),contrast=f(2**v.contrast);
  const hsl=colors.map((k,i)=>({center:centers[i],h:settings.hsl[k][0]/100*(30/360),s:settings.hsl[k][1]/100,l:settings.hsl[k][2]/100*.3})).filter(v=>v.h||v.s||v.l);
  const lum=(r,g,b)=>f(f(f(r*f(.2126))+f(g*f(.7152)))+f(b*f(.0722)));
  for(let i=0;i<before.length;i+=4){
   if(before[i+3]!==255||full[i+3]!==255)throw Error('Opaque preview required');
   let r=f(f(f(before[i]/255)*q)+f(f(full[i]/255)*s)),g=f(f(f(before[i+1]/255)*q)+f(f(full[i+1]/255)*s)),b=f(f(f(before[i+2]/255)*q)+f(f(full[i+2]/255)*s));
   if(activeBasic){
    r=f(f(r*gain[0])*tint);g=f(g*gain[1]);b=f(f(b*gain[2])*tint);
    const y=clip(lum(r,g,b)),low=f((1-y)**2),high=f(y*y),mid=f(f(4*y)*f(1-y));
    let offset=f(f(v.brightness*.35)+f(f(v.shadows*.25)*low));offset=f(offset+f(f(v.highlights*.25)*high));
    offset=f(offset+f(f(f(v.blacks*.2)*f(low**3))+f(f(v.whites*.2)*f(high**3))+f(f(f(v.brilliance*.2)*mid)*f(1-y))));
    r=f(f(f(f(r+offset)-.5)*contrast)+.5);g=f(f(f(f(g+offset)-.5)*contrast)+.5);b=f(f(f(f(b+offset)-.5)*contrast)+.5);
    const gray=lum(r,g,b),chroma=clip(f(Math.max(r,g,b)-Math.min(r,g,b))),factor=Math.max(0,f(f(1+v.saturation)+f(f(v.vibrance)*f(f(1-chroma)**2))));
    r=clip(f(gray+f(f(r-gray)*factor)));g=clip(f(gray+f(f(g-gray)*factor)));b=clip(f(gray+f(f(b-gray)*factor)));
   }
   if(activeHsl){
    const mx=Math.max(r,g,b),mn=Math.min(r,g,b),d=f(mx-mn),l=f(f(mx+mn)*.5);
    let sat=d>1e-7&&l>0&&l<1?f(d/f(1-Math.abs(f(2*l)-1))):0,h=0;
    if(d>1e-7){h=mx===r?((f(f(g-b)/d)%6)+6)%6:mx===g?f(f(f(b-r)/d)+2):f(f(f(r-g)/d)+4);h=f(h/6);}
    let dh=0,ds=0,dl=0;
    for(const region of hsl){
     let dist=Math.abs(f((((f(f(h-region.center)+.5)%1)+1)%1)-.5));const weight=f(Math.max(0,f(1-f(dist/f(50/360))))**2);
     dh=f(dh+f(weight*f(region.h)));ds=f(ds+f(weight*f(region.s)));dl=f(dl+f(weight*f(region.l)));
    }
    const hue=f(h+dh),ss=clip(f(sat*f(1+ds))),ll=clip(f(l+f(dl*Math.min(1,f(sat*5))))),c=f(f(1-Math.abs(f(2*ll)-1))*ss);
    const hp=f((((hue%1)+1)%1)*6),x=f(c*f(1-Math.abs(f(hp%2)-1))),m=f(ll-f(c/2));
    switch(Math.floor(hp)%6){case 0:r=c;g=x;b=0;break;case 1:r=x;g=c;b=0;break;case 2:r=0;g=c;b=x;break;case 3:r=0;g=x;b=c;break;case 4:r=x;g=0;b=c;break;default:r=c;g=0;b=x;}
    r=f(r+m);g=f(g+m);b=f(b+m);
   }
   if(activeCurves){r=curve(curve(r,curves.rgb),curves.r);g=curve(curve(g,curves.rgb),curves.g);b=curve(curve(b,curves.rgb),curves.b);}
   out[i]=f(clip(r)*255);out[i+1]=f(clip(g)*255);out[i+2]=f(clip(b)*255);out[i+3]=255;
  }return out;
 }
 if(typeof module!=='undefined'&&module.exports)module.exports={manualPixels,normalize};else root.SpatialManualPixels=manualPixels;
})(typeof window==='undefined'?globalThis:window);
