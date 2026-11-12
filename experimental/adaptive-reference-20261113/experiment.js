'use strict';
// The original app owns request generations, manual edits, exports and restore.
// Changing the method uses that same invalidation path, without a second UI state.
document.getElementById('engine').addEventListener('change',()=>{
  invalidate('调色方法已切换，正在重新分析。');
  autoAnalyze();
});
async function modelState(){
  try{
    const result=await api('/api/engine-state');
    document.getElementById('engine-state').textContent=result.message;
    document.querySelector('#engine option[value="model"]').disabled=result.state!=='ready';
    if(result.state==='loading')setTimeout(modelState,1200);
  }catch(error){
    document.getElementById('engine-state').textContent='无法读取模型状态，将重试。';
    setTimeout(modelState,2000);
  }
}
modelState();
