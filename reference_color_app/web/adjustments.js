'use strict';
function imageCoordinates(e,image){
  const r=image.getBoundingClientRect();
  if(!image.naturalWidth||!image.naturalHeight)return null;
  const scale=Math.min(r.width/image.naturalWidth,r.height/image.naturalHeight),w=image.naturalWidth*scale,h=image.naturalHeight*scale;
  const x=(e.clientX-r.left-(r.width-w)/2)/w,y=(e.clientY-r.top-(r.height-h)/2)/h;
  return !Number.isFinite(x)||!Number.isFinite(y)||x<0||x>1||y<0||y>1?null:{x,y};
}
// User controls are independent of reference analysis. Backend validates all state.
window.manualControls = (() => {
  const $=id=>document.getElementById(id);
  const basic=[['temperature','色温','冷','暖'],['tint','色调','绿','洋红'],['saturation','饱和度','低','高'],['vibrance','自然饱和度','低','高'],['brightness','亮度','暗','亮'],['contrast','对比度','柔和','强烈'],['highlights','高光','压低','提亮'],['shadows','阴影','压低','提亮'],['whites','白色','压低','提亮'],['blacks','黑色','压低','提亮'],['brilliance','光感','减弱','增强']];
  const colors=['red','orange','yellow','green','cyan','blue','purple','magenta'];
  const colorNames=['红','橙','黄','绿','青','蓝','紫','洋红'];
  const colorHex=['#ec5656','#e89942','#e0d14b','#5bbb59','#45c7c7','#4f96e6','#9b66d9','#d85cc0'];
  const fresh=()=>({version:1,basic:{enabled:true,...Object.fromEntries(basic.map(([k])=>[k,0]))},hsl:{enabled:true,...Object.fromEntries(colors.map(k=>[k,[0,0,0]]))},curves:{enabled:true,...Object.fromEntries(['rgb','r','g','b'].map(k=>[k,[[0,0],[1,1]]]))}});
  let state=fresh(),selectedColor='red',selectedChannel='rgb',selectedPoint=null,dragging=false,picking=false,sample=null;
  const snapshot=()=>JSON.parse(JSON.stringify(state));
  const changed=()=>{sample=null;$('curve-sample-info').textContent='参数已改变，请重新点击画面取色。';drawCurve();updateSummary();window.dispatchEvent(new Event('manual-change'));};
  function count(){
    let n=state.basic.enabled?basic.filter(([k])=>state.basic[k]!==0).length:0;
    if(state.hsl.enabled)n+=colors.filter(k=>state.hsl[k].some(v=>v!==0)).length;
    if(state.curves.enabled)n+=['rgb','r','g','b'].filter(k=>JSON.stringify(state.curves[k])!=='[[0,0],[1,1]]').length;
    return n;
  }
  function updateSummary(){ $('manual-summary').textContent=count()?`手动微调 · ${count()} 项生效`:'手动微调 · 默认原值'; }
  function row(id,label,left,right,read,write){
    const node=document.createElement('div');node.className='adjust-row';
    node.innerHTML=`<label for="${id}">${label}</label><input id="${id}" type="range" min="-100" max="100" step="1" value="0" aria-valuetext="0"><input id="${id}-number" type="number" min="-100" max="100" step="1" value="0" aria-label="${label}数值"><button type="button" class="row-reset" aria-label="重置${label}" title="重置${label}">↶</button><div class="range-hints"><span>${left}</span><span>${right}</span></div>`;
    const range=node.querySelector('input[type=range]'),number=node.querySelector('input[type=number]');
    const sync=()=>{range.value=read();number.value=read();range.setAttribute('aria-valuetext',String(read()));};
    const set=value=>{write(value);sync();changed();};
    range.addEventListener('input',()=>set(Number(range.value)));
    number.addEventListener('input',()=>{if(number.value!==''&&Number.isFinite(number.valueAsNumber))set(Math.max(-100,Math.min(100,Math.round(number.valueAsNumber))));});
    number.addEventListener('change',sync);number.addEventListener('blur',sync);
    node.querySelector('button').addEventListener('click',()=>set(0));
    node.sync=sync;return node;
  }
  const rows=[];
  function setupGroup(group){
    const check=$(group+'-enabled');check.addEventListener('change',()=>{state[group].enabled=check.checked;sync();changed();});
    $(group+'-reset').addEventListener('click',()=>{state[group]=fresh()[group];if(group==='curves')selectedPoint=null;sync();changed();});
  }
  function sync(){
    rows.forEach(r=>r.sync());
    for(const g of ['basic','hsl','curves']){
      $(g+'-enabled').checked=state[g].enabled;
      $(g+'-body').querySelectorAll('input,button').forEach(el=>{el.disabled=!state[g].enabled;});
      $(g+'-body').classList.toggle('group-disabled',!state[g].enabled);
    }
    document.querySelectorAll('[data-color]').forEach(b=>{b.classList.toggle('active',b.dataset.color===selectedColor);b.setAttribute('aria-pressed',String(b.dataset.color===selectedColor));});
    document.querySelectorAll('[data-channel]').forEach(b=>{b.classList.toggle('active',b.dataset.channel===selectedChannel);b.setAttribute('aria-pressed',String(b.dataset.channel===selectedChannel));});
    $('hsl-selected').textContent=colorNames[colors.indexOf(selectedColor)]+'色范围';
    drawCurve();updateSummary();
  }
  basic.forEach(([key,label,left,right],i)=>{
    const r=row('adjust-'+key,label,left,right,()=>state.basic[key],v=>state.basic[key]=v);
    rows.push(r);$(i<4?'basic-color-rows':'basic-tone-rows').append(r);
  });
  colors.forEach((key,i)=>{
    const b=document.createElement('button');b.type='button';b.className='color-choice';b.dataset.color=key;b.title=colorNames[i]+'色';b.setAttribute('aria-label',colorNames[i]+'色范围');b.style.setProperty('--swatch',colorHex[i]);
    b.addEventListener('click',()=>{selectedColor=key;sync();});$('hsl-colors').append(b);
  });
  ['色相','饱和度','亮度'].forEach((name,i)=>{
    const r=row('hsl-'+i,name,i===0?'偏移 −':'低',i===0?'偏移 ＋':'高',()=>state.hsl[selectedColor][i],v=>state.hsl[selectedColor][i]=v);
    rows.push(r);$('hsl-rows').append(r);
  });
  for(const group of ['basic','hsl','curves'])setupGroup(group);
  document.querySelectorAll('[data-adjust-tab]').forEach(button=>button.addEventListener('click',()=>{
    document.querySelectorAll('[data-adjust-tab]').forEach(b=>{const active=b===button;b.classList.toggle('selected',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
    for(const id of ['basic','hsl','curves'])$('panel-'+id).hidden=id!==button.dataset.adjustTab;
    document.querySelector('.adjustments').scrollTop=0;
    if(button.dataset.adjustTab!=='curves'){picking=false;document.body.classList.remove('curve-picking');$('curve-pick').setAttribute('aria-pressed','false');}
  }));
  const tabButtons=[...document.querySelectorAll('[data-adjust-tab]')];
  tabButtons.forEach((b,i)=>b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const n=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;tabButtons[n].click();tabButtons[n].focus();}}));
  document.querySelectorAll('[data-channel]').forEach(b=>b.addEventListener('click',()=>{selectedChannel=b.dataset.channel;selectedPoint=null;sync();}));
  const svg=$('curve-editor'),NS='http://www.w3.org/2000/svg';
  function drawCurve(){
    const points=state.curves[selectedChannel];const color={rgb:'#ededed',r:'#ed6666',g:'#6dc67f',b:'#61a1ef'}[selectedChannel];
    const sx=x=>16+x*288,sy=y=>204-y*188;
    svg.replaceChildren();
    for(let i=0;i<=4;i++){
      for(const attrs of [{x1:16+i*72,y1:16,x2:16+i*72,y2:204},{x1:16,y1:16+i*47,x2:304,y2:16+i*47}]){
        const line=document.createElementNS(NS,'line');Object.entries(attrs).forEach(([k,v])=>line.setAttribute(k,v));line.setAttribute('stroke','#454545');svg.append(line);
      }
    }
    const path=document.createElementNS(NS,'polyline');path.setAttribute('points',points.map(([x,y])=>`${sx(x)},${sy(y)}`).join(' '));path.setAttribute('fill','none');path.setAttribute('stroke',color);path.setAttribute('stroke-width','2');svg.append(path);
    points.forEach(([x,y],i)=>{const c=document.createElementNS(NS,'circle');c.setAttribute('cx',sx(x));c.setAttribute('cy',sy(y));c.setAttribute('r',i===selectedPoint?6:4);c.setAttribute('fill',color);c.dataset.point=i;svg.append(c);});
    if(sample){
      const values=selectedChannel==='rgb'?sample.rgb:[sample.channels[['r','g','b'].indexOf(selectedChannel)]];
      const interpolate=x=>{for(let i=1;i<points.length;i++)if(x<=points[i][0]){const a=points[i-1],b=points[i];return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0]);}return points.at(-1)[1];};
      values.forEach((x,i)=>{const c=document.createElementNS(NS,'circle');c.setAttribute('cx',sx(x));c.setAttribute('cy',sy(interpolate(x)));c.setAttribute('r','7');c.setAttribute('fill','none');c.setAttribute('stroke',selectedChannel==='rgb'?['#ed6666','#6dc67f','#61a1ef'][i]:'#00d4d9');c.setAttribute('stroke-width','2');c.style.pointerEvents='none';svg.append(c);});
    }
    const p=points[selectedPoint];$('curve-x').value=p?Math.round(p[0]*1000)/1000:'';$('curve-y').value=p?Math.round(p[1]*1000)/1000:'';
    $('curve-x').disabled=!state.curves.enabled||!p||selectedPoint===0||selectedPoint===points.length-1;
    $('curve-y').disabled=!state.curves.enabled||!p;
    $('curve-delete').disabled=!state.curves.enabled||!p||selectedPoint===0||selectedPoint===points.length-1;
  }
  function position(e){const r=svg.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width*320/288-16/288)),Math.max(0,Math.min(1,(204-(e.clientY-r.top)/r.height*220)/188))];}
  function movePoint(x,y){
    const points=state.curves[selectedChannel];if(selectedPoint===null)return;
    x=selectedPoint===0?0:selectedPoint===points.length-1?1:Math.max(points[selectedPoint-1][0]+.002,Math.min(points[selectedPoint+1][0]-.002,x));
    points[selectedPoint]=[x,Math.max(0,Math.min(1,y))];drawCurve();changed();
  }
  svg.addEventListener('pointerdown',e=>{
    if(!state.curves.enabled||e.button!==0)return;e.preventDefault();const [x,y]=position(e),points=state.curves[selectedChannel];
    let nearest=-1,best=.065;
    points.forEach(([px,py],i)=>{const d=Math.hypot(px-x,py-y);if(d<best){best=d;nearest=i;}});
    if(nearest<0){if(points.length>=16||points.some(p=>Math.abs(p[0]-x)<.004))return;points.push([x,y]);points.sort((a,b)=>a[0]-b[0]);nearest=points.findIndex(p=>p[0]===x);changed();}
    selectedPoint=nearest;dragging=true;svg.setPointerCapture(e.pointerId);drawCurve();
  });
  svg.addEventListener('pointermove',e=>{if(dragging)movePoint(...position(e));});
  const end=()=>{dragging=false;};svg.addEventListener('pointerup',end);svg.addEventListener('pointercancel',end);svg.addEventListener('lostpointercapture',end);
  $('curve-x').addEventListener('change',()=>{if(Number.isFinite($('curve-x').valueAsNumber))movePoint($('curve-x').valueAsNumber,state.curves[selectedChannel][selectedPoint][1]);});
  $('curve-y').addEventListener('change',()=>{if(Number.isFinite($('curve-y').valueAsNumber))movePoint(state.curves[selectedChannel][selectedPoint][0],$('curve-y').valueAsNumber);});
  $('curve-delete').addEventListener('click',()=>{if(selectedPoint>0&&selectedPoint<state.curves[selectedChannel].length-1){state.curves[selectedChannel].splice(selectedPoint,1);selectedPoint=null;drawCurve();changed();}});
  $('curve-channel-reset').addEventListener('click',()=>{state.curves[selectedChannel]=[[0,0],[1,1]];selectedPoint=null;drawCurve();changed();});
  $('manual-reset').addEventListener('click',()=>{state=fresh();selectedPoint=null;sync();changed();});
  $('curve-pick').addEventListener('click',()=>{picking=!picking;document.body.classList.toggle('curve-picking',picking);$('curve-pick').setAttribute('aria-pressed',String(picking));$('curve-sample-info').textContent=picking?'请点击画面内像素；黑边不参与取色。':'取色已关闭，标记不改变曲线。';});
  for(const id of ['before','after'])$(id).addEventListener('click',e=>{
    if(!picking||!state.curves.enabled)return;
    const point=imageCoordinates(e,$(id));if(!point)return;
    window.dispatchEvent(new CustomEvent('curve-pick',{detail:point}));
  });
  sync();
  return {snapshot,count,clearSample(){sample=null;$('curve-sample-info').textContent='点击取色后，在画面内选择像素；仅定位，不改曲线。';drawCurve();},setSample(value){sample=value;$('curve-sample-info').textContent=`原图像素 (${value.pixel.join(', ')}) · 曲线输入 RGB ${value.rgb.map(v=>Math.round(v*255)).join(' / ')} · 仅定位`;drawCurve();},restore(value){const defaults=fresh();state=defaults;if(value){for(const group of ['basic','hsl','curves'])if(value[group])Object.assign(state[group],JSON.parse(JSON.stringify(value[group])));}selectedPoint=null;sync();}};
})();
