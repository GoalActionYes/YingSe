'use strict';
// UI acceptance against our own isolated app, not automation of a user's desktop.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const [url, fixtures, out, executablePath, modulePath] = process.argv.slice(2);
const {chromium} = require(modulePath);

async function main() {
  const browser = await chromium.launch({executablePath, headless:true});
  let context,page;
  const report = {browser:browser.version(), page_errors:[], failed_requests:[]};
  const settled = () => page.waitForFunction(() => state.mapping && !state.previewRunning && !state.previewNext && fastStrength.basis && !fastStrength.pending && !document.getElementById('after').hidden, null, {timeout:30000});
  const decoded = () => page.evaluate(async () => {await document.getElementById('after').decode();});
  async function exportFile(kind) {
    await page.selectOption('#format',kind);
    const settings=await page.evaluate(()=>({mapping_id:state.mapping,media_id:state.target.id,strength:effectiveStrength(),adjustments:controls.snapshot()}));
    await page.click('#export');
    await page.locator('#result-links').waitFor({state:'visible',timeout:60000});
    const href = await page.locator('#result-link').getAttribute('href');
    const response = await context.request.get(url+href);
    assert.equal(response.status(),200);
    const bytes = await response.body();
    assert(bytes.length>100);
    fs.writeFileSync(path.join(out,'browser-export.'+kind),bytes);
    return {bytes:bytes.length,download_link_visible:true,settings};
  }
  try {
    context = await browser.newContext({viewport:{width:1440,height:1000},deviceScaleFactor:1});
    page = await context.newPage();
    page.on('pageerror', error => report.page_errors.push(error.message));
    page.on('requestfailed', request => report.failed_requests.push({path:new URL(request.url()).pathname,error:request.failure()?.errorText}));
    await page.goto(url,{waitUntil:'networkidle'});
    assert.match(await page.title(),/自适应/);
    await page.setInputFiles('#target-file',path.join(fixtures,'target.png'));
    await page.waitForFunction(() => state.target && state.uploading===0);
    await page.setInputFiles('#reference-file',path.join(fixtures,'reference.png'));
    await settled(); await decoded();
    report.photo_import={width:await page.locator('#after').evaluate(im=>im.naturalWidth),height:await page.locator('#after').evaluate(im=>im.naturalHeight)};
    await page.screenshot({path:path.join(out,'photo.png'),fullPage:true});

    // Deliberately hold preview HTTP responses: a successful local paint must precede them.
    await page.route('**/api/preview',async route=>{await new Promise(resolve=>setTimeout(resolve,1200));await route.continue();});
    async function changeAndMeasure(id,value) {
      return page.evaluate(async ({id,value}) => {
        const image=document.getElementById('after'),old=image.src,start=performance.now();
        const painted=new Promise((resolve,reject)=>{
          const timeout=setTimeout(()=>{observer.disconnect();reject(Error('No decoded preview within 1000ms: '+JSON.stringify({id,value,eligible:fastStrength.eligible(),basis:!!fastStrength.basis,current:fastStrength.current(),key:fastStrength.key,serverSignature:fastStrength.serverSignature,pending:!!fastStrength.pending,frame:fastStrength.frame})));},1000);
          const observer=new MutationObserver(async()=>{
            if(image.src===old)return;
            observer.disconnect();
            try {await image.decode();await new Promise(requestAnimationFrame);clearTimeout(timeout);resolve(performance.now()-start);}
            catch(error){clearTimeout(timeout);reject(error);}
          });observer.observe(image,{attributes:true,attributeFilter:['src']});
        });
        const input=document.getElementById(id);input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));
        return painted;
      },{id,value});
    }
    const strength=report.strength_local_decoded_ms=[];
    for(const value of [0,10,20,30,40,50,60,70,80,90,100,70,40,10]) strength.push(await changeAndMeasure('strength',value));
    await settled();
    report.strength_local_decoded_ms=strength;
    const manual=report.manual_local_decoded_ms=[];
    for(const value of [5,10,15,20,25,30,20,10,0]) manual.push(await changeAndMeasure('adjust-temperature',value));
    await settled();
    report.manual_local_decoded_ms=manual;
    report.preview_http_delay_ms=1200;
    await page.unroute('**/api/preview');
    const matches=await page.evaluate(async()=>{
      const result=await api('/api/preview',{mapping_id:state.mapping,strength:effectiveStrength(),adjustments:controls.snapshot(),interactive:true});
      return result.after===document.getElementById('after').src;
    });
    assert(matches,'Latest UI preview does not match the final server render');
    report.latest_preview_matches_server=true;

    await page.locator('#adjust-brightness-number').fill('18');
    await page.locator('#adjust-brightness-number').blur();
    await settled();
    await page.click('#manual-undo');
    assert.equal(await page.inputValue('#adjust-brightness-number'),'0');
    await page.click('#manual-redo');
    assert.equal(await page.inputValue('#adjust-brightness-number'),'18');
    await settled();
    report.undo_redo=true;
    await page.click('#view-compare');
    assert.equal(await page.locator('#comparison').getAttribute('data-view'),'compare');
    await page.click('#tab-curves');
    const curvesBefore=await page.evaluate(()=>JSON.stringify(controls.snapshot().curves));
    await page.click('#curve-pick');
    await page.locator('#after').click();
    await page.waitForFunction(()=>document.getElementById('curve-sample-info').textContent.includes('原图像素'));
    assert.equal(await page.evaluate(()=>JSON.stringify(controls.snapshot().curves)),curvesBefore);
    report.curve_sample={text:await page.textContent('#curve-sample-info'),does_not_edit_curve:true};
    await page.screenshot({path:path.join(out,'curve-sample.png'),fullPage:true});
    report.png_export=await exportFile('png');
    await page.waitForFunction(()=>document.getElementById('save-state').textContent.includes('已保存'));
    await page.reload({waitUntil:'networkidle'});
    await settled();
    assert.equal(await page.inputValue('#adjust-brightness-number'),'18');
    assert.equal(await page.inputValue('#strength'),'10');
    report.project_restore=true;

    await page.setInputFiles('#target-file',path.join(fixtures,'input-audio.mp4'));
    await settled(); await decoded();
    assert.equal(await page.inputValue('#format'),'mp4');
    await page.locator('#frame-time').fill('1');
    await page.click('#frame-load');
    await settled(); await decoded();
    report.video_import_and_frame_select=true;
    await page.screenshot({path:path.join(out,'video.png'),fullPage:true});
    report.mp4_export=await exportFile('mp4');
    assert.deepEqual(report.page_errors,[]);
    assert.deepEqual(report.failed_requests,[]);
    report.passed=true;
  } catch(error) {
    report.failure=error.stack;
    report.status=page?await page.locator('#status').textContent().catch(()=>null):null;
    if(page)await page.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});
    throw error;
  } finally {
    try {fs.writeFileSync(path.join(out,'browser-results.json'),JSON.stringify(report,null,2));}
    finally {try {if(context)await context.close();}finally {await browser.close();}}
  }
}
process.stdin.once('data',()=>{
  process.stdin.pause();
  main().catch(error=>{console.error(error.stack);process.exitCode=1;});
});
