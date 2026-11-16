/* Isolated automated form checks. These votes are never human evidence. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const [root, modulePath, executablePath] = process.argv.slice(2);
const {chromium} = require(modulePath);

(async () => {
  const browser = await chromium.launch({executablePath, headless:true});
  try {
    const context = await browser.newContext({acceptDownloads:true, viewport:{width:1280,height:1000}});
    await context.setOffline(true);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    const checks = [];
    for (let part = 1; part <= 2; part++) {
      const file = path.join(root, 'part-'+part, '匿名评价.html');
      const html = fs.readFileSync(file, 'utf8');
      for (const forbidden of ['current13_multiref_epoch6','failed_expanded4096_epoch4','private-key']) assert(!html.includes(forbidden));
      await page.goto(pathToFileURL(file).href);
      assert((await page.textContent('#progress')).includes('0 / 32'));
      await page.waitForFunction(() => document.querySelector('#panel').complete && document.querySelector('#panel').naturalWidth === 1024);
      await page.screenshot({path:path.join(root,'empty-part-'+part+'.png'), fullPage:true});
      await page.click('#export');
      assert((await page.textContent('#message')).includes('请填写'));
      await page.fill('#reviewer','AUTOMATED TEST');
      for (const key of ['style','naturalness','coordination']) await page.locator('input[name='+key+'][value=A]').check();
      async function exportData() {
        const pending = page.waitForEvent('download');
        await page.click('#export');
        const downloaded = await pending;
        return JSON.parse(fs.readFileSync(await downloaded.path(),'utf8'));
      }
      const partial = await exportData();
      assert.equal(partial.complete,false);
      assert.equal(partial.completed_cases,1);
      assert.equal(partial.reviewer,'AUTOMATED TEST');
      for (let i=1; i<32; i++) {
        await page.click('#next');
        for (const key of ['style','naturalness','coordination']) await page.locator('input[name='+key+'][value=unsure]').check();
      }
      const complete = await exportData();
      assert.equal(complete.completed_cases,32);
      assert.equal(complete.complete,true);
      assert.equal(complete.ratings[31].coordination,'unsure');
      const manifest = fs.readFileSync(path.join(root,'part-'+part,'manifest.json'));
      assert.equal(complete.material_sha256,require('node:crypto').createHash('sha256').update(manifest).digest('hex'));
      await page.reload();
      assert((await page.textContent('#progress')).includes('32 / 32'));
      checks.push({part, initial_votes:0, partial:1, complete:32, unsure_preserved:true, restored:true});
    }
    assert.equal(errors.length,0);
    await context.close();
    const fresh = await browser.newContext();
    const untouched = await fresh.newPage();
    await untouched.goto(pathToFileURL(path.join(root,'part-1','匿名评价.html')).href);
    assert((await untouched.textContent('#progress')).includes('0 / 32'));
    await fresh.close();
    const report = {scope:'Automated isolated browser checks only; zero human votes',passed:true,checks,page_storage_isolated:true,fresh_context_empty:true,errors};
    fs.writeFileSync(path.join(root,'browser-checks.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report));
  } finally {
    await browser.close();
  }
})().catch(e => {console.error(e.stack);process.exitCode=1;});
