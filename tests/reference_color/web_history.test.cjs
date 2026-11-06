const {test}=require('node:test');const assert=require('node:assert/strict');
const {createHistory}=require('../../reference_color_app/web/history.js');
test('gesture is one undo, redo branches and snapshots isolated',()=>{
 const h=createHistory({value:0});h.begin();h.record({value:1});h.record({value:30});h.end();
 assert.equal(h.undo().value,0);assert.equal(h.redo().value,30);
 h.undo();h.record({value:5});assert.equal(h.canRedo(),false);
 const v=h.undo();v.value=100;assert.equal(h.redo().value,5);
});
test('bounded history and restore establish a fresh baseline',()=>{
 const h=createHistory({value:0},3);for(let i=1;i<=10;i++)h.record({value:i});
 for(let i=0;i<3;i++)h.undo();assert.equal(h.canUndo(),false);
 h.reset({value:42});assert.equal(h.canUndo(),false);assert.equal(h.canRedo(),false);
 h.record({value:42});assert.equal(h.canUndo(),false);
});
test('returning to start or ending an empty gesture creates no phantom step',()=>{
 const h=createHistory({v:0});h.begin();h.record({v:30});h.record({v:0});h.end();assert.equal(h.canUndo(),false);
 h.begin();h.end();h.record({v:10});h.record({v:20});assert.equal(h.undo().v,10);assert.equal(h.undo().v,0);
});
